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
	t.equals(#Rules.engineArguments(settings()), 0, 'no --backend when no engine is set')
	t.equals(#Rules.engineArguments(settings({ engine = '' })), 0)
	t.equals(#Rules.engineArguments({}), 0, 'a missing pref means not set')
end)

t.test('each engine name reaches the command line as --backend', function()
	for _, engine in ipairs(ENGINES) do
		local args = Rules.engineArguments(settings({ engine = engine }))
		t.isNotNil(args, engine .. ' was refused')
		t.equals(args[1], '--backend', engine .. ': the flag')
		t.equals(args[2], engine, engine .. ': the value')
		t.equals(#args, 2, engine .. ': exactly the flag and the value')
	end
end)

t.test('an unknown engine is refused with the four choices named', function()
	local args, message = Rules.engineArguments(settings({ engine = 'anthropic' }))
	t.isNil(args, 'an unknown engine was passed on to the command line')
	t.isNotNil(message, 'no message for the unknown engine')
	t.isNotNil(string.find(message, 'anthropic', 1, true), 'the message does not name what was set')
	for _, engine in ipairs(ENGINES) do
		t.isNotNil(string.find(message, engine, 1, true), 'the message does not name ' .. engine)
	end
	t.isNil(Rules.engineArguments(settings({ engine = 'MLX' })), 'names are the exact words')
	t.isNil(Rules.engineArguments(settings({ engine = 42 })), 'a junk pref is refused, not crashed on')
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
