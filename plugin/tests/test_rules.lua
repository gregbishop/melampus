--[[
Tests for the write-decision rules.

These encode CLAUDE.md §5.3's non-negotiables. They are written against a pure
module with no Lightroom dependencies precisely so they can run here, in a second,
rather than only being discoverable by damaging a real catalog:

  * never overwrite an existing user rating, flag, label or keyword
  * dry-run writes nothing
  * a second run is a no-op
  * auto-reject is off unless explicitly enabled

`photo` in these tests is a plain table describing existing catalog state, and
the rules return a plan describing intended changes. Nothing here touches a
catalog; applying the plan is the thin Lightroom layer's job.
--]]

local t = require('harness')
local mock = require('lrmock')
local Rules = require('MelampusRules')

local function result(overrides)
	local r = {
		file = '0A1A2475.jpg',
		status = 'ok',
		species = 'Tricolored Heron',
		scientificName = 'Egretta tricolor',
		confidence = 0.95,
		burstAgreement = 1.0,
		rangeFlag = false,
		taxon = 'bird',
		alternates = 'Little Blue Heron',
		abstain = false,
		-- Ratings derive from the quality composite, never from ID confidence:
		-- how sure the model is about a species says nothing about the photograph.
		quality = 78,
		model = 'qwen3-vl-30b',
		encounter = 3,
	}
	for k, v in pairs(overrides or {}) do r[k] = v end
	return r
end

local function photo(overrides)
	local p = {
		rating = nil, pickStatus = 0, colorNameForLabel = nil,
		keywords = {}, melampus = {},
	}
	for k, v in pairs(overrides or {}) do p[k] = v end
	return p
end

local function settings(overrides)
	local s = Rules.defaultSettings()
	for k, v in pairs(overrides or {}) do s[k] = v end
	return s
end

-- ── the safety rules ───────────────────────────────────────────────────────
t.test('defaults are safe', function()
	local s = Rules.defaultSettings()
	t.isTrue(s.dryRun, 'dry-run must default ON')
	t.isFalse(s.autoReject, 'auto-reject must default OFF')
	t.isFalse(s.overwriteRating, 'rating overwrite must default OFF')
	t.isFalse(s.overwriteLabel, 'label overwrite must default OFF')
	t.isFalse(s.overwriteKeywords, 'keyword overwrite must default OFF')
end)

t.test('an existing user rating is never overwritten by default', function()
	local plan = Rules.planFor(result(), photo({ rating = 4 }), settings({ writeRating = true }))
	t.isNil(plan.rating, 'clobbered a rating the user had already set')
	t.contains(plan.skipped, 'rating: already set')
end)

t.test('a rating is written when the field is empty', function()
	local plan = Rules.planFor(result(), photo(), settings({ writeRating = true }))
	t.isNotNil(plan.rating, 'refused to write into an empty rating')
end)

t.test('overwriting a rating requires explicit opt-in', function()
	local plan = Rules.planFor(result(), photo({ rating = 4 }),
		settings({ writeRating = true, overwriteRating = true }))
	t.isNotNil(plan.rating, 'explicit overwrite opt-in was ignored')
end)

t.test('an existing colour label is never overwritten by default', function()
	local plan = Rules.planFor(result(), photo({ colorNameForLabel = 'red' }),
		settings({ writeLabel = true }))
	t.isNil(plan.label)
	t.contains(plan.skipped, 'label: already set')
end)

t.test('an existing pick or reject flag is never touched by default', function()
	local plan = Rules.planFor(result(), photo({ pickStatus = 1 }), settings({ writeFlags = true }))
	t.isNil(plan.pickStatus, 'overwrote a flag the user had set')
end)

t.test('keywords already on the photo are not duplicated', function()
	local existing = { 'Tricolored Heron' }
	local plan = Rules.planFor(result(), photo({ keywords = existing }), settings())
	for _, kw in ipairs(plan.keywords) do
		t.isFalse(kw == existing[1], 'proposed a keyword the photo already carries')
	end
end)

t.test('auto-reject never fires unless enabled', function()
	local bad = result({ confidence = 0.05, burstAgreement = 0.2 })
	local plan = Rules.planFor(bad, photo(), settings({ writeFlags = true }))
	t.isFalse(plan.pickStatus == -1, 'rejected a photo with auto-reject off')
end)

t.test('auto-reject fires only when enabled and the photo is empty', function()
	local bad = result({ confidence = 0.05, burstAgreement = 0.2, quality = 5 })
	local plan = Rules.planFor(bad, photo(),
		settings({ writeFlags = true, autoReject = true, rejectBelowQuality = 20 }))
	t.equals(plan.pickStatus, -1)
end)

-- ── the confidence gate ────────────────────────────────────────────────────
-- ── keyword shape ──────────────────────────────────────────────────────────
-- Flat by default. A hierarchy is harder to read in the keyword panel, and
-- confidence and taxon are bookkeeping that belongs in the metadata panel, not
-- in a keyword list the photographer has to live with for years.
t.test('a confident identification writes just the species name', function()
	local plan = Rules.planFor(result(), photo(), settings())
	t.contains(plan.keywords, 'Tricolored Heron')
	t.equals(#plan.keywords, 1, 'wrote more than the species name: '
		.. table.concat(plan.keywords, ', '))
end)

t.test('no confidence or taxon keywords are ever written', function()
	local plan = Rules.planFor(result(), photo(), settings())
	for _, kw in ipairs(plan.keywords) do
		t.isFalse(kw == 'High' or kw == 'Bird' or string.find(kw, 'Confidence', 1, true) ~= nil,
			'bookkeeping leaked into the keyword list: ' .. kw)
	end
end)

t.test('a low-confidence result gets a review marker and no species', function()
	local plan = Rules.planFor(result({ confidence = 0.4 }), photo(), settings())
	t.contains(plan.keywords, 'Needs ID')
	for _, kw in ipairs(plan.keywords) do
		t.isFalse(kw == 'Tricolored Heron', 'named a species below the confidence gate')
	end
end)

t.test('an unstable burst blocks the species keyword even at high confidence', function()
	local plan = Rules.planFor(result({ burstAgreement = 0.4 }), photo(), settings())
	for _, kw in ipairs(plan.keywords) do
		t.isFalse(kw == 'Tricolored Heron',
			'trusted a confident ID the model contradicted across the burst')
	end
end)

t.test('a range-flagged result is flagged and never auto-tagged', function()
	local plan = Rules.planFor(result({ rangeFlag = true }), photo(), settings())
	t.contains(plan.keywords, 'Out of Range')
	for _, kw in ipairs(plan.keywords) do
		t.isFalse(kw == 'Tricolored Heron',
			'auto-tagged a species that does not occur at this location')
	end
end)

t.test('an abstention writes no species and says why', function()
	local plan = Rules.planFor(result({ abstain = true, species = nil }), photo(), settings())
	t.contains(plan.keywords, 'Needs ID')
	t.equals(plan.metadata.species, nil)
end)

t.test('hierarchical style remains available for those who want it', function()
	local plan = Rules.planFor(result(), photo(), settings({ keywordStyle = 'hierarchical' }))
	t.contains(plan.keywords, 'Melampus > Species > Tricolored Heron')
end)

-- ── behaviour ──────────────────────────────────────────────────────────────
t.test('behaviour becomes keywords', function()
	local plan = Rules.planFor(result({ behaviour = { 'in-flight', 'feeding' } }),
		photo(), settings())
	t.contains(plan.keywords, 'In-Flight')
	t.contains(plan.keywords, 'Feeding')
end)

t.test('behaviour is written even when the species is not', function()
	-- A bird too blurred to name can still plainly be in flight.
	local plan = Rules.planFor(result({ confidence = 0.2, behaviour = { 'in-flight' } }),
		photo(), settings())
	t.contains(plan.keywords, 'Needs ID')
	t.contains(plan.keywords, 'In-Flight')
end)

t.test('age is off by default and skips non-observations', function()
	local plan = Rules.planFor(result({ ageSex = 'adult male' }), photo(), settings())
	for _, kw in ipairs(plan.keywords) do
		t.isFalse(kw == 'Adult Male', 'wrote age with the setting off')
	end
	local on = Rules.planFor(result({ ageSex = 'adult male' }), photo(),
		settings({ writeAgeSex = true }))
	t.contains(on.keywords, 'Adult Male')

	-- The model sometimes echoes the whole menu back rather than observing.
	local echoed = Rules.planFor(
		result({ ageSex = 'adult male | juvenile | breeding plumage | indeterminate' }),
		photo(), settings({ writeAgeSex = true }))
	for _, kw in ipairs(echoed.keywords) do
		t.isFalse(string.find(kw, '|', 1, true) ~= nil, 'turned an echoed menu into a keyword')
	end

	local vague = Rules.planFor(result({ ageSex = 'indeterminate' }), photo(),
		settings({ writeAgeSex = true }))
	for _, kw in ipairs(vague.keywords) do
		t.isFalse(kw == 'Indeterminate', '"indeterminate" is not worth a keyword')
	end
end)

-- ── batch size ─────────────────────────────────────────────────────────────
t.test('batch size is clamped to something sane', function()
	t.equals(Rules.batchSize(settings()), 25, 'default should be 25')
	t.equals(Rules.batchSize(settings({ analyzeBatchSize = 10 })), 10)
	t.equals(Rules.batchSize(settings({ analyzeBatchSize = 0 })), 1,
		'zero would stall the run')
	t.equals(Rules.batchSize(settings({ analyzeBatchSize = -5 })), 1)
	t.equals(Rules.batchSize(settings({ analyzeBatchSize = 99999 })), 500,
		'an enormous batch defeats incremental updates entirely')
	t.equals(Rules.batchSize(settings({ analyzeBatchSize = 'nonsense' })), 25,
		'a junk pref should fall back, not crash')
	t.equals(Rules.batchSize({}), 25)
end)

-- ── the engine preference (card #403) ──────────────────────────────────────
-- Where inference runs is the user's choice. The preference is named engine,
-- its values are the owner's words, and the plugin passes it to the CLI as
-- --backend. Unset means no --backend at all: the CLI's own default applies
-- (mlx today; the first engine that can run here once card #404 detects).
local ENGINES = { 'mlx', 'ollama', 'openai', 'claude' }

t.test('the engines are exactly the four words, in order', function()
	t.equals(#Rules.ENGINES, #ENGINES, 'not the four engines')
	for i, engine in ipairs(ENGINES) do
		t.equals(Rules.ENGINES[i], engine, 'engine ' .. i)
	end
end)

t.test('the default engine is unset, so the CLI picks', function()
	t.equals(Rules.defaultSettings().engine, '', 'the default must mean "not set"')
	local engine, message = Rules.chosenEngine(settings())
	t.isNil(engine, 'an engine was chosen when none is set')
	t.isNil(message, 'no engine set is not a mistake')
	t.isNil(Rules.chosenEngine(settings({ engine = '' })), 'empty means not set')
	t.isNil(Rules.chosenEngine({}), 'a missing pref means not set')
end)

t.test('each engine name is the engine chosen', function()
	for _, engine in ipairs(ENGINES) do
		t.equals(Rules.chosenEngine(settings({ engine = engine })), engine, engine .. ' was refused')
	end
end)

t.test('an unknown engine is refused with the four choices named', function()
	local engine, message = Rules.chosenEngine(settings({ engine = 'anthropic' }))
	t.isNil(engine, 'an unknown engine was passed on to the command line')
	t.isNotNil(message, 'no message for the unknown engine')
	t.isNotNil(string.find(message, 'anthropic', 1, true), 'the message does not name what was set')
	for _, engine in ipairs(ENGINES) do
		t.isNotNil(string.find(message, engine, 1, true), 'the message does not name ' .. engine)
	end
	t.isNil(Rules.chosenEngine(settings({ engine = 'MLX' })), 'names are the exact words')
	local _, why = Rules.chosenEngine(settings({ engine = 'MLX' }))
	t.isNotNil(why, 'a near miss is refused with a message, not silently unset')
	t.isNil(Rules.chosenEngine(settings({ engine = 42 })), 'a junk pref is refused, not crashed on')
end)

-- ── detection to picker items (card #405) ─────────────────────────────────
-- The CLI decides what can run here (--detect-engines, card #404); the dialog
-- only shows it. Rules.engineItems turns the decoded verdict list into the
-- picker's items, in the owner's order, with the unavailable ones disabled, and
-- a note to show under the picker that carries their reasons.
local verdicts = mock.detectionVerdicts

--- The picker's items indexed by value, so a test can name one: byValue.ollama.
local function itemsByValue(items)
	local byValue = {}
	for _, item in ipairs(items) do byValue[item.value] = item end
	return byValue
end

t.test('the picker lists the four engines in the owner\'s order, after letting Melampus choose', function()
	local items = Rules.engineItems(verdicts())
	t.equals(items[1].value, '', 'the first item must be the unset preference: let the CLI choose')
	t.isTrue(items[1].enabled, 'letting Melampus choose is always allowed')
	t.equals(#items, 5, 'the automatic item and the four engines')
	for i, engine in ipairs(ENGINES) do
		t.equals(items[i + 1].value, engine, 'item ' .. (i + 1))
		t.isNotNil(items[i + 1].title, engine .. ' has no title')
	end
end)

t.test('unavailable engines are disabled, and the note carries the reason detection gave', function()
	-- The dialog reads enabled and link from an item and shows the reasons
	-- from the note under the picker; an item carries nothing the dialog
	-- does not read.
	local items, note = Rules.engineItems(verdicts({ mlx = { available = false, reason = 'needs Apple Silicon' } }))
	local byValue = itemsByValue(items)
	t.isFalse(byValue.mlx.enabled, 'mlx should be greyed')
	t.isFalse(byValue.ollama.enabled, 'ollama should be greyed')
	for _, item in ipairs(items) do
		t.isNil(item.reason, item.value .. ' carries a reason nothing reads; the note has it')
	end
	t.isTrue(byValue.openai.enabled, 'openai is available')
	t.isTrue(byValue.claude.enabled, 'claude is available')
	-- The title says so too: per-item enabled is not visible outside
	-- Lightroom, the title is.
	for _, item in ipairs(items) do
		local saysNotAvailable = string.find(item.title, ' (not available)', 1, true) ~= nil
		local endsWithIt = string.sub(item.title, -#' (not available)') == ' (not available)'
		if item.enabled then
			t.isFalse(saysNotAvailable, item.value .. ' is available but its title says otherwise: ' .. item.title)
		else
			t.isTrue(endsWithIt, item.value .. ' is greyed but its title does not end with "(not available)": ' .. item.title)
		end
	end
	t.isNotNil(string.find(note, 'needs Apple Silicon', 1, true), 'the note does not carry the mlx reason:\n' .. note)
	t.isNotNil(string.find(note, 'no Ollama server', 1, true), 'the note does not carry the ollama reason:\n' .. note)
	t.isNil(string.find(note, 'API key required', 1, true), 'the note explains available engines:\n' .. note)
end)

t.test('the link is the ollama item\'s alone, from the address its reason names', function()
	-- Only Ollama is something to go and install (Done-when 3): another
	-- engine's reason stays text in the note, address and all.
	local items = Rules.engineItems(verdicts())
	local byValue = itemsByValue(items)
	t.equals(byValue.ollama.link, 'https://ollama.com/download')
	t.isNil(byValue.mlx.link, 'mlx is not ollama, so it gets no link')
	t.isNil(byValue.openai.link, 'openai is not ollama, so it gets no link')
	local MLX_ADDRESS = 'https://example.com/apple-silicon'
	local note
	items, note = Rules.engineItems(verdicts({ mlx = { available = false, reason = 'needs Apple Silicon; see ' .. MLX_ADDRESS } }))
	byValue = itemsByValue(items)
	t.isNil(byValue.mlx.link, 'an address in the mlx reason must not become a link: only ollama\'s does')
	t.isNotNil(string.find(note, MLX_ADDRESS, 1, true), 'the note does not carry the mlx address as text:\n' .. note)
	local answering = verdicts({ ollama = { available = true, reason = 'Ollama is answering at http://127.0.0.1:11434' } })
	items = Rules.engineItems(answering)
	byValue = itemsByValue(items)
	t.isTrue(byValue.ollama.enabled)
	t.isNil(byValue.ollama.link, 'an available engine needs no link')
end)

t.test('without verdicts nothing is greyed and the note says why', function()
	local problem = 'Melampus could not find its analysis program.'
	local items, note = Rules.engineItems(nil, problem)
	t.equals(#items, 5)
	for _, item in ipairs(items) do
		t.isTrue(item.enabled, item.value .. ' was greyed with no verdict to grey it')
		t.isNil(item.link)
	end
	for i, engine in ipairs(ENGINES) do t.equals(items[i + 1].value, engine) end
	t.equals(note, problem)
	-- Junk from the executable is the same case, and never a crash.
	items, note = Rules.engineItems({ 'not', 'verdicts' }, problem)
	t.equals(#items, 5)
	t.isTrue(items[3].enabled)
	t.equals(note, problem)
	items, note = Rules.engineItems({})
	t.equals(#items, 5)
	t.equals(note, '', 'nothing to say when there are no verdicts and no problem')
end)

t.test('when every engine is available the note is empty', function()
	local _, note = Rules.engineItems(verdicts({ ollama = { available = true, reason = 'Ollama is answering' } }))
	t.equals(note, '')
end)

t.test('each cloud engine names the variable its key travels in; local engines none', function()
	t.equals(Rules.keyVariable('openai'), 'MELAMPUS_OPENAI_KEY')
	t.equals(Rules.keyVariable('claude'), 'MELAMPUS_ANTHROPIC_KEY')
	t.isNil(Rules.keyVariable('mlx'))
	t.isNil(Rules.keyVariable('ollama'))
	t.isNil(Rules.keyVariable(''))
	t.isNil(Rules.keyVariable(nil))
end)

-- ── the model download's protocol and button (card #408) ───────────────────
-- The executable prints `progress <done> <total>`, `done <path>` and
-- `cancelled` (docs/config.md § Downloading the model); the plugin reads the
-- last of them from a file. The sample lines are shared with the Python test
-- of download.Update so the two parsers cannot drift.
local function sampleLines()
	local here = debug.getinfo(1, 'S').source:match('^@(.*)[/\\]') or '.'
	local path = here .. '/../../service/tests/fixtures/download-lines.txt'
	local rows = {}
	for row in io.lines(path) do
		if row:sub(1, 1) ~= '#' then
			local fields = {}
			for field in (row .. '\t'):gmatch('([^\t]*)\t') do fields[#fields + 1] = field end
			rows[#rows + 1] = fields
		end
	end
	assert(#rows >= 10, 'the sample lines were not read from ' .. path)
	return rows
end

t.test('the download line parser agrees with the Python one on every shared sample line', function()
	local seen = { progress = 0, done = 0, cancelled = 0, rejected = 0 }
	for _, row in ipairs(sampleLines()) do
		local line, state = row[1], row[2]
		local update = Rules.parseDownloadLine(line)
		seen[state] = seen[state] + 1
		if state == 'rejected' then
			t.isNil(update, 'accepted a line that is not an update: ' .. line)
		else
			t.isNotNil(update, 'rejected ' .. line)
			t.equals(update.state, state, line)
			if state == 'progress' then
				t.equals(update.bytesDone, tonumber(row[3]), line)
				t.equals(update.bytesTotal, tonumber(row[4]), line)
			elseif state == 'done' then
				t.equals(update.path, row[3], line)
			end
			local withNewline = Rules.parseDownloadLine(line .. '\r\n')
			t.equals(withNewline.state, state, 'a line read with its line ending: ' .. line)
			t.equals(withNewline.path, update.path)
		end
	end
	for _, state in ipairs({ 'progress', 'done', 'cancelled', 'rejected' }) do
		t.isTrue(seen[state] > 0, 'no sample line of kind ' .. state)
	end
end)

t.test('the latest update in the progress file is the last line that parses', function()
	local text = 'progress 0 100\nprogress 40 100\nprogress 70 100\n'
	local latest = Rules.latestDownloadUpdate(text)
	t.equals(latest.state, 'progress')
	t.equals(latest.bytesDone, 70)
	t.equals(latest.bytesTotal, 100)
	latest = Rules.latestDownloadUpdate(text .. 'done /hf/hub/models--x--y/snapshots/abc\n')
	t.equals(latest.state, 'done')
	t.equals(latest.path, '/hf/hub/models--x--y/snapshots/abc')
	-- A line still being written is not an update yet; the previous one stands.
	latest = Rules.latestDownloadUpdate(text .. 'progress 80')
	t.equals(latest.bytesDone, 70)
	t.isNil(Rules.latestDownloadUpdate(''), 'nothing yet')
	t.isNil(Rules.latestDownloadUpdate(nil), 'no file yet')
	t.isNil(Rules.latestDownloadUpdate('warning: something on stderr\n'))
end)

t.test('the download button names the model and its size, or says the size is unknown', function()
	local repo = 'mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit'
	t.equals(Rules.downloadTitle({ repo = repo, bytes_total = 18300000000 }),
		'Download ' .. repo .. ' (18.3 GB)')
	t.equals(Rules.downloadTitle({ repo = repo, bytes_total = 734000000 }),
		'Download ' .. repo .. ' (734 MB)')
	-- JSON null decodes to nil: the hub could not be reached.
	t.equals(Rules.downloadTitle({ repo = repo }), 'Download ' .. repo .. ' (size unknown)')
end)

t.test('the engines with a model to download are mlx and ollama, in the owner\'s order (card #409)', function()
	t.equals(table.concat(Rules.MODEL_ENGINES, ','), 'mlx,ollama')
	-- The same title for Ollama's model: its name from the status, and its
	-- size once Ollama holds it; 'size unknown' before, since Ollama's list
	-- gives sizes for held models only.
	t.equals(Rules.downloadTitle({ repo = 'qwen3-vl:8b-instruct', bytes_total = 6100000000 }),
		'Download qwen3-vl:8b-instruct (6.1 GB)')
	t.equals(Rules.downloadTitle({ repo = 'qwen3-vl:8b-instruct' }), 'Download qwen3-vl:8b-instruct (size unknown)')
end)

t.test('progress reads as bytes of the total and a portion between 0 and 1', function()
	local text, portion = Rules.downloadProgress({ state = 'progress', bytesDone = 3100000000, bytesTotal = 18300000000 })
	t.equals(text, '3.1 GB of 18.3 GB')
	t.isTrue(math.abs(portion - 3100000000 / 18300000000) < 1e-9)
	text, portion = Rules.downloadProgress({ state = 'progress', bytesDone = 0, bytesTotal = 0 })
	t.equals(portion, 0, 'no division by zero before the total is known')
end)

t.test('the tail of a log is its last eight lines; a shorter one passes through whole', function()
	-- What a failure message shows of the CLI log: enough to name the
	-- cause, not the whole run.
	local lines = {}
	for i = 1, 20 do lines[i] = 'line ' .. i end
	local text = table.concat(lines, '\n') .. '\n'
	local kept = {}
	for line in string.gmatch(Rules.tail(text), '[^\n]+') do kept[#kept + 1] = line end
	t.equals(#kept, 8, 'lines kept')
	t.equals(kept[1], 'line 13')
	t.equals(kept[8], 'line 20')
	t.equals(Rules.tail(table.concat(lines, '\n')) .. '\n', Rules.tail(text),
		'a log cut off mid-line keeps the same eight lines as one ending in a line break')
	t.equals(Rules.tail('one\ntwo\nthree\n'), 'one\ntwo\nthree\n', 'fewer lines pass through whole')
	t.equals(Rules.tail('no newline at the end'), 'no newline at the end')
	t.equals(Rules.tail(''), '')
	t.equals(Rules.tail(nil), '', 'no log file yet')
end)

t.test('the engine the picker resolves to is the picked one, else the first detection says can run', function()
	t.equals(Rules.resolvedEngine('ollama', verdicts()), 'ollama')
	t.equals(Rules.resolvedEngine('', verdicts()), 'mlx', 'the default on an Apple Silicon Mac is mlx')
	t.equals(Rules.resolvedEngine(nil, verdicts()), 'mlx')
	t.equals(Rules.resolvedEngine('', verdicts({ mlx = { available = false, reason = 'needs Apple Silicon' } })),
		'openai', 'the first available in the owner\'s order')
	t.isNil(Rules.resolvedEngine('', nil), 'without detection nothing is resolved')
end)

t.test('whether an engine can run here is detection\'s verdict on it, and nothing runs without detection', function()
	-- The dialog asks this to decide whether to ask about the MLX model at
	-- all; it holds no engine knowledge of its own.
	t.isTrue(Rules.canRun(verdicts(), 'mlx'))
	t.isFalse(Rules.canRun(verdicts(), 'ollama'), 'no Ollama server is answering')
	t.isFalse(Rules.canRun(verdicts({ mlx = { available = false, reason = 'needs Apple Silicon' } }), 'mlx'))
	t.isFalse(Rules.canRun(nil, 'mlx'), 'without detection nothing is known to run')
	t.isFalse(Rules.canRun({ 'not', 'verdicts' }, 'mlx'), 'output that is not the list')
	t.isFalse(Rules.canRun(verdicts(), 'scripted'), 'an engine detection never names')
end)

-- ── colour labels ──────────────────────────────────────────────────────────
t.test('colour labels mean something specific', function()
	local s = settings({ writeLabel = true })
	t.equals(Rules.planFor(result(), photo(), s).label, 'green',
		'a confident identification should be green')
	t.equals(Rules.planFor(result({ confidence = 0.3 }), photo(), s).label, 'yellow',
		'an unsure identification should be yellow')
	t.equals(Rules.planFor(result({ rangeFlag = true }), photo(), s).label, 'red',
		'an out-of-range species should be red, not lost among the unsure')
end)

t.test('an existing colour label is still never overwritten', function()
	local plan = Rules.planFor(result({ rangeFlag = true }), photo({ colorNameForLabel = 'blue' }),
		settings({ writeLabel = true }))
	t.isNil(plan.label, 'clobbered a label the user had set')
end)

-- ── idempotency ────────────────────────────────────────────────────────────
t.test('a second run over unchanged state proposes nothing', function()
	local s = settings({ writeRating = true })
	local first = Rules.planFor(result(), photo(), s)

	local after = photo({
		rating = first.rating,
		keywords = first.keywords,
		melampus = first.metadata,
	})
	local second = Rules.planFor(result(), after, s)

	t.isTrue(Rules.isEmpty(second), 'a re-run proposed changes it had already made')
end)

t.test('force makes a re-run write again', function()
	local s = settings({ writeRating = true, force = true })
	local first = Rules.planFor(result(), photo(), settings({ writeRating = true }))
	local after = photo({ rating = first.rating, keywords = first.keywords, melampus = first.metadata })
	t.isFalse(Rules.isEmpty(Rules.planFor(result(), after, s)), 'force did not re-apply')
end)

-- ── dry run ────────────────────────────────────────────────────────────────
t.test('dry run marks the plan as not to be applied', function()
	t.isFalse(Rules.shouldApply(settings({ dryRun = true })), 'dry run would have written')
	t.isTrue(Rules.shouldApply(settings({ dryRun = false })))
end)

-- ── robustness against bad input ───────────────────────────────────────────
t.test('a failed result produces no writes at all', function()
	local plan = Rules.planFor({ file = 'x.jpg', status = 'unprocessed' }, photo(), settings())
	t.isTrue(Rules.isEmpty(plan), 'wrote metadata for an unprocessed photo')
end)

t.test('missing fields do not raise', function()
	local ok = pcall(Rules.planFor, { file = 'x.jpg', status = 'ok' }, photo(), settings())
	t.isTrue(ok, 'a sparse record raised instead of degrading')
end)

t.test('keyword text is sanitised', function()
	local plan = Rules.planFor(result({ species = 'Heron > Weird | Name' }), photo(), settings())
	for _, kw in ipairs(plan.keywords) do
		t.isFalse(string.find(kw, '>', 1, true) ~= nil,
			'species name injected a hierarchy separator: ' .. kw)
	end
end)

return t.summary()
