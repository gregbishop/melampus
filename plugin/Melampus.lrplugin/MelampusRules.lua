--[[
Write-decision rules — pure logic, no Lightroom dependencies.

Everything that decides *what* to change lives here so it can be tested outside
Lightroom in a second, rather than being discoverable only by damaging a real
catalog. The Lightroom layer's job is narrow: read existing state into a plain
table, call planFor, and apply the returned plan inside a write transaction.

The rules implement CLAUDE.md §5.3's non-negotiables. Two are worth stating
plainly because they drive most of the code below:

  * Nothing the user set is ever overwritten unless overwrite is explicitly
    enabled for that field type. Default is write-only-where-empty.
  * A species keyword is written only when the identification clears every gate.
    A wrong keyword is worse than no keyword, so the failure mode is a review
    marker, never a guess.
--]]

local Rules = {}

Rules.SCHEMA_VERSION = '1'
Rules.KEYWORD_ROOT = 'Melampus'

function Rules.defaultSettings()
	return {
		-- Safety. All three of these are load-bearing.
		dryRun = true,           -- first run against any catalog writes nothing
		autoReject = false,      -- an automated reject pass feels destructive
		force = false,           -- re-runs are a no-op unless forced

		-- Per-field overwrite permission. Off means write only where empty.
		overwriteRating = false,
		overwriteLabel = false,
		overwriteFlags = false,
		overwriteKeywords = false,

		-- What to write at all. Ratings and labels come from the quality
		-- composite, which is Stage 2 work, so they stay off until it exists.
		writeKeywords = true,
		-- 'flat' writes just the species name. A hierarchy is harder to read in
		-- the keyword panel, and taxon and confidence are bookkeeping that
		-- already live in the metadata panel — they do not belong in a keyword
		-- list you have to live with for years.
		keywordStyle = 'flat',
		-- Behaviour is already extracted for every photo and was being thrown
		-- away. "Find my in-flight shots" is a question a photographer actually
		-- asks, and unlike a species name it does not risk a wrong identification
		-- -- a bird either is wading or it is not.
		writeBehaviour = true,
		writeAgeSex = false,
		writeMetadata = true,
		-- Ratings now have a real source: the quality composite from §4.1.
		-- They reflect how good the photograph is, never how sure the model is
		-- about the species — those are different questions.
		writeRating = true,
		writeLabel = false,
		writeFlags = false,

		-- Gates for auto-tagging a species. Measured on the development corpus:
		-- confidence >= 0.90 was right ~82% of the time on its own, but combined
		-- with burst agreement >= 0.80 that rises sharply. Agreement is the
		-- stronger signal of the two.
		minConfidence = 0.90,
		minBurstAgreement = 0.80,
		rejectBelowQuality = 20,
		-- Below this many frames a burst is too small for ranking to be
		-- meaningful, so the absolute quality score is used instead.
		minBurstForRanking = 4,
		-- Working JPEG previews are deleted once analysis succeeds. They are
		-- roughly 78 KB each, so a whole-library sweep would otherwise leave a
		-- few hundred megabytes in a temp folder nothing ever cleans. They are
		-- always kept when analysis fails, since then they are evidence.
		keepPreviews = false,
		-- Photos analysed per batch. Each batch is written to the catalog before
		-- the next starts, so stars and keywords appear during a long run rather
		-- than all at the end. Smaller means more visible progress and slightly
		-- more overhead; 25 is about three minutes between updates.
		analyzeBatchSize = 25,
		-- What kind of shoot this is. Wildlife asks what organism is in the
		-- frame; sport asks what activity is happening. Keeping them apart stops
		-- a footballer being routed to 'mammal' and asked for a scientific name.
		profile = 'wildlife',
		-- Where inference runs (card #403): one of Rules.ENGINES, passed to the
		-- CLI as --backend. Empty means the user has not chosen, so the CLI's
		-- own default applies: mlx today, and the first engine that can run on
		-- this machine once card #404's detection lands.
		engine = '',
	}
end

--- The engines a user can choose between, in the owner's words and the
-- executable's order: the owner's four, then the two subscription CLIs
-- (card #423). These are the CLI's --backend names; the offline test fake
-- is not one. The executable's --detect-engines prints one verdict per
-- name in this order, and test_lua_plugin.py holds the two to each other.
Rules.ENGINES = { 'mlx', 'ollama', 'openai', 'claude', 'claude-code', 'codex' }

--- The engine the preference chooses: its name when it is one of
-- Rules.ENGINES, nil when none is set, so the CLI decides. An unknown value
-- returns nil and a message naming the choices, so the run stops here rather
-- than on the CLI's usage error. How the name reaches the CLI is the
-- command line's business (MelampusAnalyze.lua).
function Rules.chosenEngine(settings)
	local engine = settings and settings.engine
	if engine == nil or engine == '' then return nil end
	for _, known in ipairs(Rules.ENGINES) do
		if engine == known then return engine end
	end
	return nil, 'Melampus does not know the engine "' .. tostring(engine)
		.. '".\n\nThe engines are: ' .. table.concat(Rules.ENGINES, ', ')
		.. '.\n\nSet one of those in Settings, or leave it unset to let Melampus choose.'
end

--- The engines with a local model to fetch, and so a Download row (card
-- #408 for mlx, #409 for ollama), in the owner's order. The executable's
-- --download-model, --model-status and --remove-model act for the engine
-- passed as --backend; the others have no model to fetch.
Rules.MODEL_ENGINES = { 'mlx', 'ollama' }

--- The variable a cloud engine's API key travels in to the executable
-- (providers.KEY_VARIABLES on the Python side), which is also the name the
-- key is stored under. nil for an engine that needs no key.
Rules.KEY_VARIABLES = {
	openai = 'MELAMPUS_OPENAI_KEY',
	claude = 'MELAMPUS_ANTHROPIC_KEY',
}

function Rules.keyVariable(engine)
	return Rules.KEY_VARIABLES[engine]
end

--- The engine picker's items from what the executable said (card #405):
-- `verdicts` is the decoded JSON of --detect-engines, a list of
-- { engine, title, available, reason, install }. The first item leaves the
-- choice to the executable (the unset preference); then Rules.ENGINES in
-- order, each titled as its verdict says (the executable is the one place
-- that names an engine; card #423), disabled when detection said it cannot
-- run here.
-- A disabled item whose verdict carries an install page (`install`: where
-- to go and get the engine, when going and getting it is the fix) carries
-- that address as `link`. An address a reason merely names stays text in
-- the note.
-- Without verdicts (no executable, or output that is not the list) nothing
-- is greyed, the names stand in for the titles, and `problem` is the note.
-- Returns the items and the note to show under the picker: one line per
-- unavailable engine with its reason, or the problem. An item carries only
-- what the dialog reads: title, value, enabled, link, and the reason
-- Rules.pickedReason shows for the picked engine.
function Rules.engineItems(verdicts, problem)
	local byEngine = {}
	if type(verdicts) == 'table' then
		for _, verdict in ipairs(verdicts) do
			if type(verdict) == 'table' and type(verdict.engine) == 'string' then
				byEngine[verdict.engine] = verdict
			end
		end
	end
	local items = {
		{ title = 'Let Melampus choose — the first engine that can run here',
			value = '', enabled = true },
	}
	local lines = {}
	for _, engine in ipairs(Rules.ENGINES) do
		local verdict = byEngine[engine]
		local available = verdict == nil or verdict.available ~= false
		local reason = verdict and tostring(verdict.reason or '') or ''
		local title = verdict and type(verdict.title) == 'string' and verdict.title or engine
		local item = {
			title = title .. (available and '' or ' (not available)'),
			value = engine, enabled = available, reason = reason,
		}
		if not available then
			-- Only something to go and install gets a link, and where to go is
			-- the address the verdict carries (Ollama, card #405; the two
			-- subscription CLIs, card #423), never one scraped out of the
			-- reason: a CLI's reason is the CLI's own words, and can name the
			-- billing docs or a line the program printed on stderr, neither of
			-- them an install page (review round 9, finding 1). Whatever a
			-- reason names stays text in the note, address and all.
			-- An https address at that: the link is the one value the plugin
			-- hands the operating system's URL opener (MelampusSettings:
			-- LrHttp.openUrlInBrowser), and it arrives as JSON read back off
			-- disk, so a file: address or a registered custom handler would
			-- be launched without question (security review round 11). The
			-- scrape this field replaced matched 'https?://' and constrained
			-- the scheme by construction; the predicate is what replaces it.
			local install = verdict.install
			if type(install) == 'string' and string.match(install, '^https://') then item.link = install end
			lines[#lines + 1] = title .. ': ' .. reason
		end
		items[#items + 1] = item
	end
	local note = table.concat(lines, '\n')
	if note == '' and next(byEngine) == nil then note = problem or '' end
	return items, note
end

--- The engines detection says can run here, by name, in the order the
-- executable prints them (the owner's); empty without verdicts (no
-- executable, or output that is not the list). The one walk of the
-- verdicts for what can run.
local function availableEngines(verdicts)
	local names = {}
	for _, verdict in ipairs(type(verdicts) == 'table' and verdicts or {}) do
		if type(verdict) == 'table' and verdict.available == true then names[#names + 1] = verdict.engine end
	end
	return names
end

--- Whether detection said `engine` can run here (card #408: the dialog
-- asks about the MLX model only where mlx can run). false without
-- detection: nothing is known to run.
function Rules.canRun(verdicts, engine)
	for _, name in ipairs(availableEngines(verdicts)) do
		if name == engine then return true end
	end
	return false
end

--- What to say under the picker about the picked engine: its item's reason
-- (card #423: a signed-in subscription CLI's says what every frame bills
-- to, before a run; a cloud engine's names the key it needs), or '' when
-- nothing is picked, the value is not an item, or there was no detection.
function Rules.pickedReason(items, engine)
	for _, item in ipairs(items or {}) do
		if item.value == engine and item.value ~= '' then return item.reason or '' end
	end
	return ''
end

--- The engine a picker value comes to (card #408): the picked one, or with
-- the preference unset the first that detection says can run here, in the
-- owner's order, which is the executable's own default (providers
-- .default_engine). nil when nothing is picked and there is no detection.
function Rules.resolvedEngine(engine, verdicts)
	if engine ~= nil and engine ~= '' then return engine end
	return availableEngines(verdicts)[1]
end

-- ── the model download (card #408) ─────────────────────────────────────────
-- The executable's --download-model prints one line per update on stdout
-- (docs/config.md § Downloading the model), which the plugin redirects to a
-- file and reads back. This parser mirrors download.Update.parse on the
-- Python side, and both are tested against the same sample lines.

--- One protocol line, or nil for any other line. `progress <done> <total>`
-- gives { state = 'progress', bytesDone, bytesTotal }; `done <path>` gives
-- { state = 'done', path } with the path the rest of the line, spaces and
-- all; `cancelled` alone gives { state = 'cancelled' }.
function Rules.parseDownloadLine(line)
	if type(line) ~= 'string' then return nil end
	line = string.gsub(line, '[\r\n]+$', '')
	local word, rest = string.match(line, '^([^ ]*) (.*)$')
	if not word then word, rest = line, '' end
	if word == 'progress' then
		local done, total = string.match(rest, '^(%d+) (%d+)$')
		if done then return { state = 'progress', bytesDone = tonumber(done), bytesTotal = tonumber(total) } end
	elseif word == 'done' and rest ~= '' then
		return { state = 'done', path = rest }
	elseif word == 'cancelled' and rest == '' then
		return { state = 'cancelled' }
	end
	return nil
end

--- The last update in the text of the progress file, or nil while there is
-- none: a line still being written does not parse and the one before stands.
function Rules.latestDownloadUpdate(text)
	local latest = nil
	for line in string.gmatch(text or '', '[^\n]+') do
		latest = Rules.parseDownloadLine(line) or latest
	end
	return latest
end

--- A byte count as the button and the progress line show it.
function Rules.formatBytes(bytes)
	if type(bytes) ~= 'number' then return 'size unknown' end
	if bytes >= 1e9 then return string.format('%.1f GB', bytes / 1e9) end
	if bytes >= 1e6 then return string.format('%.0f MB', bytes / 1e6) end
	return string.format('%d bytes', bytes)
end

--- The Download button's title from the decoded --model-status JSON: the
-- model's name and its size, or 'size unknown' when the hub could not be
-- reached (bytes_total null).
function Rules.downloadTitle(status)
	return 'Download ' .. tostring(status and status.repo) .. ' (' .. Rules.formatBytes(status and status.bytes_total) .. ')'
end

--- A progress update as text ('3.1 GB of 18.3 GB') and the portion done,
-- 0 to 1, for a progress scope.
function Rules.downloadProgress(update)
	local done, total = update.bytesDone or 0, update.bytesTotal or 0
	local portion = total > 0 and done / total or 0
	return Rules.formatBytes(done) .. ' of ' .. Rules.formatBytes(total), portion
end

--- The last lines of a log, for a failure message: enough to name the
-- cause, not the whole run. A trailing line break ends the last line; it
-- does not start another. Fewer lines pass through whole; nil (no file
-- yet) is empty.
local TAIL_LINES = 8

function Rules.tail(text)
	text = text or ''
	local last = #text
	if string.sub(text, last, last) == '\n' then last = last - 1 end
	local start, count = 1, 0
	for i = last, 1, -1 do
		if string.sub(text, i, i) == '\n' then
			count = count + 1
			if count == TAIL_LINES then
				start = i + 1
				break
			end
		end
	end
	return string.sub(text, start)
end

-- Keyword hierarchy uses '>' as its separator, so a species name containing one
-- would silently create extra levels. Strip anything structural.
local function sanitise(text)
	if text == nil then return nil end
	text = tostring(text)
	text = string.gsub(text, '[>|]', ' ')
	text = string.gsub(text, '%s+', ' ')
	text = string.gsub(text, '^%s*(.-)%s*$', '%1')
	if text == '' then return nil end
	return text
end

local function titleCase(text)
	if not text then return nil end
	return (string.gsub(text, "(%a)([%w']*)", function(first, rest)
		return string.upper(first) .. rest
	end))
end

local function contains(list, needle)
	for _, value in ipairs(list or {}) do
		if value == needle then return true end
	end
	return false
end

--- Batch size, clamped. Prefs are user-editable and persist across versions, so
-- a stale or nonsensical value must not stall a run or spin it one photo at a
-- time.
function Rules.batchSize(settings)
	local size = tonumber(settings and settings.analyzeBatchSize) or 25
	size = math.floor(size)
	if size < 1 then return 1 end
	if size > 500 then return 500 end
	return size
end

function Rules.confidenceBand(confidence, settings)
	confidence = confidence or 0
	if confidence >= (settings.minConfidence or 0.9) then return 'High' end
	if confidence >= 0.7 then return 'Medium' end
	return 'Low'
end

--- Does this identification clear every gate for an automatic species keyword?
function Rules.passesGate(result, settings)
	if result.abstain then return false end
	if not result.species or result.species == '' then return false end
	if result.rangeFlag then return false end
	if (result.confidence or 0) < (settings.minConfidence or 0.9) then return false end
	-- burstAgreement may be absent for a single-frame encounter; absent is not failure.
	local agreement = result.burstAgreement
	if agreement ~= nil and agreement < (settings.minBurstAgreement or 0.8) then
		return false
	end
	return true
end

function Rules.ratingFor(result, settings)
	-- Prefer the frame's rank within its own burst. Absolute sharpness is not
	-- comparable between a smooth white egret and a patterned heron, but "best
	-- frame of this burst" is exactly the question culling asks. A lone frame
	-- has no burst to rank against, so it falls back to the absolute score.
	local rank = result.qualityRank
	local frames = result.encounterFrames or 0
	if rank ~= nil and frames >= (settings.minBurstForRanking or 4) then
		if rank >= 0.90 then return 5 end
		if rank >= 0.70 then return 4 end
		if rank >= 0.40 then return 3 end
		if rank >= 0.15 then return 2 end
		return 1
	end

	local quality = result.quality
	if quality == nil then return nil end
	-- Breakpoints are deliberately strict at the top. Five stars should mean
	-- "the best frames of the shoot", not "most of them" — a rating everything
	-- earns is useless for culling.
	if quality >= 90 then return 5 end
	if quality >= 75 then return 4 end
	if quality >= 55 then return 3 end
	if quality >= 35 then return 2 end
	return 1
end

--- Build the intended-change plan for one photo.
-- @param result   identification result for this file
-- @param photo    existing catalog state: rating, pickStatus, colorNameForLabel,
--                 keywords (list), melampus (previously written custom fields)
-- @param settings see defaultSettings
function Rules.planFor(result, photo, settings)
	result = result or {}
	photo = photo or {}
	settings = settings or Rules.defaultSettings()

	local plan = {
		file = result.file,
		keywords = {},
		metadata = {},
		skipped = {},
		rating = nil,
		label = nil,
		pickStatus = nil,
	}

	-- Nothing is written for a photo the pipeline could not process. An
	-- unprocessed photo must stay untouched rather than acquire empty fields.
	if result.status ~= nil and result.status ~= 'ok' then
		plan.skipped[#plan.skipped + 1] = 'result: ' .. tostring(result.status)
		return plan
	end

	local existingKeywords = photo.keywords or {}
	local previous = photo.melampus or {}

	-- Idempotency: if we already wrote this exact identification, propose nothing.
	if not settings.force and previous.schemaVersion == Rules.SCHEMA_VERSION then
		local sameSpecies = (previous.species or '') == (sanitise(result.species) or '')
		local sameModel = (previous.model or '') == (result.model or '')
		if sameSpecies and sameModel then
			plan.skipped[#plan.skipped + 1] = 'already processed'
			return plan
		end
	end

	local species = sanitise(result.species)
	local passes = Rules.passesGate(result, settings)

	-- ── keywords ───────────────────────────────────────────────────────────
	if settings.writeKeywords then
		local proposed = {}
		local root = Rules.KEYWORD_ROOT
		local hierarchical = settings.keywordStyle == 'hierarchical'

		if hierarchical then
			if result.taxon and result.taxon ~= '' and result.taxon ~= 'none' then
				proposed[#proposed + 1] = root .. ' > Taxon > ' .. titleCase(sanitise(result.taxon))
			end
			if passes then
				proposed[#proposed + 1] = root .. ' > Species > ' .. species
				proposed[#proposed + 1] = root .. ' > Confidence > '
					.. Rules.confidenceBand(result.confidence, settings)
			else
				proposed[#proposed + 1] = root .. ' > Review > Needs ID'
			end
			if result.rangeFlag then
				proposed[#proposed + 1] = root .. ' > Notable > Out of range'
			end
		else
			-- Flat: the species name, and nothing else unless it needs attention.
			if passes then
				proposed[#proposed + 1] = species
			else
				proposed[#proposed + 1] = 'Needs ID'
			end
			if result.rangeFlag then
				proposed[#proposed + 1] = 'Out of Range'
			end
		end

		-- Behaviour and age are written regardless of the species gate. They do
		-- not depend on getting the species right: a bird that cannot be named
		-- can still plainly be in flight.
		if settings.writeBehaviour and type(result.behaviour) == 'table' then
			for _, behaviour in ipairs(result.behaviour) do
				local clean = sanitise(behaviour)
				if clean then proposed[#proposed + 1] = titleCase(clean) end
			end
		end
		if settings.writeAgeSex then
			local age = sanitise(result.ageSex)
			-- The model sometimes echoes the whole menu of options back; that is
			-- not an observation and must not become a keyword.
			if age and age ~= 'indeterminate' and not string.find(age, '|', 1, true) then
				proposed[#proposed + 1] = titleCase(age)
			end
		end

		for _, keyword in ipairs(proposed) do
			if contains(existingKeywords, keyword) then
				plan.skipped[#plan.skipped + 1] = 'keyword exists: ' .. keyword
			else
				plan.keywords[#plan.keywords + 1] = keyword
			end
		end
	end

	-- ── custom metadata ────────────────────────────────────────────────────
	-- Always recorded, including for gated-out results: the panel needs to show
	-- what the model thought even when we declined to act on it.
	if settings.writeMetadata then
		plan.metadata = {
			species = passes and species or nil,
			scientificName = sanitise(result.scientificName),
			alternates = sanitise(result.alternates),
			taxon = sanitise(result.taxon),
			confidence = result.confidence and string.format('%.2f', result.confidence) or nil,
			burstAgreement = result.burstAgreement
				and string.format('%.2f', result.burstAgreement) or nil,
			rangeFlag = result.rangeFlag and 'out-of-range' or 'in-range',
			encounter = result.encounter and tostring(result.encounter) or nil,
			model = result.model,
			schemaVersion = Rules.SCHEMA_VERSION,
			verdict = 'unreviewed',
		}
	end

	-- ── rating ─────────────────────────────────────────────────────────────
	if settings.writeRating then
		local rating = Rules.ratingFor(result, settings)
		if rating == nil then
			plan.skipped[#plan.skipped + 1] = 'rating: no quality score'
		elseif photo.rating ~= nil and photo.rating ~= 0 and not settings.overwriteRating then
			plan.skipped[#plan.skipped + 1] = 'rating: already set'
		else
			plan.rating = rating
		end
	end

	-- ── colour label ───────────────────────────────────────────────────────
	if settings.writeLabel then
		if photo.colorNameForLabel and photo.colorNameForLabel ~= ''
			and not settings.overwriteLabel then
			plan.skipped[#plan.skipped + 1] = 'label: already set'
		else
			-- green  = confident identification, species keyword written
			-- red    = species named that does not occur here; the notable pile,
			--          holding both model errors and genuinely unusual records
			-- yellow = needs a look, no species claimed
			if result.rangeFlag then
				plan.label = 'red'
			elseif passes then
				plan.label = 'green'
			else
				plan.label = 'yellow'
			end
		end
	end

	-- ── flags ──────────────────────────────────────────────────────────────
	if settings.writeFlags then
		local userSetFlag = photo.pickStatus ~= nil and photo.pickStatus ~= 0
		if userSetFlag and not settings.overwriteFlags then
			plan.skipped[#plan.skipped + 1] = 'flag: already set'
		elseif settings.autoReject and result.quality
			and result.quality < (settings.rejectBelowQuality or 20) then
			plan.pickStatus = -1
		end
	end

	return plan
end

function Rules.isEmpty(plan)
	if plan.rating ~= nil or plan.label ~= nil or plan.pickStatus ~= nil then return false end
	if plan.keywords and #plan.keywords > 0 then return false end
	if plan.metadata then
		for _ in pairs(plan.metadata) do return false end
	end
	return true
end

function Rules.shouldApply(settings)
	return settings.dryRun == false
end

return Rules
