--[[
Runs the real MelampusImport.lua against a mock Lightroom SDK.

This is the test that was missing. Everything before it checked decision logic
in isolation; this executes the actual file the plugin loads, with a real
results JSON on disk, and asserts on what landed in the catalog.

It is what would have caught the two failures seen in practice: keywords silently
failing to attach, and a counter reporting success for photos that got nothing.
--]]

local t = require('harness')
local mock = require('lrmock')

local PLUGIN = os.getenv('MELAMPUS_PLUGIN')
	or '../Melampus.lrplugin'

local function writeResults(path, records)
	local parts = {}
	for _, r in ipairs(records) do
		local cands = {}
		for _, c in ipairs(r.candidates or {}) do
			cands[#cands + 1] = string.format(
				'{"common_name":%q,"scientific_name":%q,"confidence":%s,"reasoning":""}',
				c[1], c[2] or '', tostring(c[3]))
		end
		parts[#parts + 1] = string.format(
			'{"file":%q,"status":%q,"model":"test","identification":'
			.. '{"taxon":"bird","candidates":[%s],"age_sex":"adult","count":1,'
			.. '"behavior":[],"diagnostic_features_visible":true,"abstain":%s,'
			.. '"abstain_reason":null},"burst_agreement":%s,"range_flag":%s,'
			.. '"encounter":%d,"quality":%s}',
			r.file, r.status or 'ok', table.concat(cands, ','),
			tostring(r.abstain or false), tostring(r.agreement or 1.0),
			tostring(r.rangeFlag or false), r.encounter or 1, tostring(r.quality or 80))
	end
	local handle = assert(io.open(path, 'w'))
	handle:write('[', table.concat(parts, ',\n'), ']')
	handle:close()
end

local RESULTS = os.tmpname() .. '.json'

--- Run the real import file end to end and hand back the resulting state.
local function runImport(records, photos, prefs)
	writeResults(RESULTS, records)
	mock.reset({ prefs = prefs or {}, confirmAnswer = 'ok' })
	mock.state.prefs.resultsPath = RESULTS
	for _, spec in ipairs(photos) do
		mock.addPhoto(spec[1], spec[2] or {})
	end
	mock.install(PLUGIN)
	package.loaded['MelampusJson'] = nil
	package.loaded['MelampusRules'] = nil
	package.loaded['MelampusLog'] = nil
	local chunk = assert(loadfile(PLUGIN .. '/MelampusImport.lua'))
	local ok, err = pcall(chunk)
	return ok, err
end

local function defaultPrefs(extra)
	package.loaded['MelampusRules'] = nil
	local Rules = dofile(PLUGIN .. '/MelampusRules.lua')
	local prefs = Rules.defaultSettings()
	prefs.dryRun = false
	for k, v in pairs(extra or {}) do prefs[k] = v end
	return prefs
end

-- ── the plugin actually runs ───────────────────────────────────────────────
t.test('the real import file executes without error', function()
	local ok, err = runImport(
		{ { file = '0A1A2475.jpg', candidates = { { 'Tricolored Heron', 'Egretta tricolor', 0.95 } } } },
		{ { '0A1A2475.CR3' } }, defaultPrefs())
	t.isTrue(ok, 'import raised: ' .. tostring(err))
end)

-- ── keywords genuinely attach ──────────────────────────────────────────────
t.test('a confident identification attaches a species keyword to the photo', function()
	runImport(
		{ { file = '0A1A2475.jpg', candidates = { { 'Tricolored Heron', 'Egretta tricolor', 0.95 } } } },
		{ { '0A1A2475.CR3' } }, defaultPrefs())
	local paths = mock.state.photos[1]:keywordPaths()
	t.contains(paths, 'Tricolored Heron',
		'no species keyword attached; got: ' .. table.concat(paths, ', '))
end)

t.test('keywords land on every photo, not just the first', function()
	local records, photos = {}, {}
	for i = 1, 25 do
		records[#records + 1] = { file = string.format('f%02d.jpg', i),
			candidates = { { 'Snowy Egret', 'Egretta thula', 0.95 } } }
		photos[#photos + 1] = { string.format('f%02d.CR3', i) }
	end
	runImport(records, photos, defaultPrefs())
	local missing = 0
	for _, photo in ipairs(mock.state.photos) do
		if #photo:keywordPaths() == 0 then missing = missing + 1 end
	end
	t.equals(missing, 0, missing .. ' of 25 photos received no keyword at all')
end)

t.test('repeated species do not collapse into a single tagged photo', function()
	-- The real bug: name collisions made createKeyword return nil for every
	-- photo after the first, so only one photo ended up tagged.
	local records, photos = {}, {}
	for i = 1, 10 do
		records[#records + 1] = { file = string.format('g%02d.jpg', i),
			candidates = { { 'Green Heron', 'Butorides virescens', 0.95 } } }
		photos[#photos + 1] = { string.format('g%02d.CR3', i) }
	end
	runImport(records, photos, defaultPrefs())
	local tagged = 0
	for _, photo in ipairs(mock.state.photos) do
		for _, path in ipairs(photo:keywordPaths()) do
			if path == 'Green Heron' then tagged = tagged + 1 end
		end
	end
	t.equals(tagged, 10, 'only ' .. tagged .. ' of 10 photos got the shared keyword')
end)

t.test('hierarchical keywords nest correctly across many photos', function()
	-- This is the case that actually broke. With a multi-segment path,
	-- createKeyword returning nil for a colliding name reset the parent to nil
	-- and dumped 'Species' and 'Taxon' at the root. Flat keywords are single
	-- segment so they cannot expose it; this test can.
	local records, photos = {}, {}
	for i = 1, 8 do
		records[#records + 1] = { file = string.format('h%02d.jpg', i),
			candidates = { { 'Tricolored Heron', 'Egretta tricolor', 0.95 } } }
		photos[#photos + 1] = { string.format('h%02d.CR3', i) }
	end
	runImport(records, photos, defaultPrefs({ keywordStyle = 'hierarchical' }))

	local tagged = 0
	for _, photo in ipairs(mock.state.photos) do
		for _, path in ipairs(photo:keywordPaths()) do
			if path == 'Melampus > Species > Tricolored Heron' then tagged = tagged + 1 end
		end
	end
	t.equals(tagged, 8, 'only ' .. tagged .. ' of 8 photos got the nested species keyword')

	-- And nothing structural may end up at the root.
	for _, kw in ipairs(mock.catalog:getKeywords()) do
		t.isFalse(kw.name == 'Species' or kw.name == 'Taxon' or kw.name == 'Confidence',
			'"' .. kw.name .. '" was created at the root instead of under Melampus')
	end
end)

-- ── ratings ────────────────────────────────────────────────────────────────
t.test('a star rating is written from the quality score', function()
	runImport(
		{ { file = 'r1.jpg', quality = 95,
			candidates = { { 'Anhinga', 'Anhinga anhinga', 0.95 } } } },
		{ { 'r1.CR3' } }, defaultPrefs())
	t.equals(mock.state.photos[1]:getRawMetadata('rating'), 5,
		'quality 95 did not produce 5 stars')
end)

t.test('an existing rating is not overwritten', function()
	runImport(
		{ { file = 'r2.jpg', quality = 95,
			candidates = { { 'Anhinga', 'Anhinga anhinga', 0.95 } } } },
		{ { 'r2.CR3', { rating = 2 } } }, defaultPrefs())
	t.equals(mock.state.photos[1]:getRawMetadata('rating'), 2, 'clobbered a user rating')
end)

-- ── the gates ──────────────────────────────────────────────────────────────
t.test('a low-confidence identification gets Needs ID, not a species', function()
	runImport(
		{ { file = 'n1.jpg', candidates = { { 'Little Blue Heron', 'Egretta caerulea', 0.4 } } } },
		{ { 'n1.CR3' } }, defaultPrefs())
	local paths = mock.state.photos[1]:keywordPaths()
	t.contains(paths, 'Needs ID')
	for _, p in ipairs(paths) do
		t.isFalse(p == 'Little Blue Heron', 'named a species below the gate')
	end
end)

t.test('an out-of-range identification is flagged and not named', function()
	runImport(
		{ { file = 'o1.jpg', rangeFlag = true,
			candidates = { { 'Long-tailed Cuckoo', 'Cuculus micropterus', 0.95 } } } },
		{ { 'o1.CR3' } }, defaultPrefs())
	local paths = mock.state.photos[1]:keywordPaths()
	t.contains(paths, 'Out of Range')
	for _, p in ipairs(paths) do
		t.isFalse(p == 'Long-tailed Cuckoo', 'tagged a species that does not occur here')
	end
end)

-- ── dry run ────────────────────────────────────────────────────────────────
t.test('preview mode writes nothing when declined', function()
	mock.reset()
	local prefs = defaultPrefs({ dryRun = true })
	writeResults(RESULTS, { { file = 'd1.jpg',
		candidates = { { 'Snowy Egret', 'Egretta thula', 0.95 } } } })
	mock.reset({ prefs = prefs, confirmAnswer = 'cancel' })
	mock.state.prefs.resultsPath = RESULTS
	mock.addPhoto('d1.CR3', {})
	mock.install(PLUGIN)
	package.loaded['MelampusJson'] = nil; package.loaded['MelampusRules'] = nil
	package.loaded['MelampusLog'] = nil
	assert(loadfile(PLUGIN .. '/MelampusImport.lua'))()
	t.equals(#mock.state.photos[1]:keywordPaths(), 0, 'preview wrote despite being declined')
	t.equals(#mock.state.writeTransactions, 0, 'preview opened a write transaction')
end)

-- ── matching and safety ────────────────────────────────────────────────────
t.test('jpg results match CR3 photos by basename', function()
	runImport(
		{ { file = '0A1A9999.jpg', candidates = { { 'Anhinga', 'Anhinga anhinga', 0.95 } } } },
		{ { '0A1A9999.CR3' } }, defaultPrefs())
	t.isTrue(#mock.state.photos[1]:keywordPaths() > 0, 'basename matching failed')
end)

t.test('photos with no result are left completely untouched', function()
	runImport(
		{ { file = 'known.jpg', candidates = { { 'Anhinga', 'Anhinga anhinga', 0.95 } } } },
		{ { 'known.CR3' }, { 'stranger.CR3' } }, defaultPrefs())
	t.equals(#mock.state.photos[2]:keywordPaths(), 0, 'touched a photo with no result')
	t.isNil(mock.state.photos[2]:getRawMetadata('rating'))
end)

t.test('no file reading happens inside a write transaction', function()
	-- Yielding inside withWriteAccessDo is what produces the classic
	-- "yielding is not allowed" error, and file I/O yields.
	runImport(
		{ { file = 'y1.jpg', candidates = { { 'Anhinga', 'Anhinga anhinga', 0.95 } } } },
		{ { 'y1.CR3' } }, defaultPrefs())
	t.isFalse(mock.state.yieldInsideWrite, 'read a file inside a write gate')
end)

t.test('a second run over the same photos changes nothing', function()
	local records = { { file = 'i1.jpg', quality = 95,
		candidates = { { 'Anhinga', 'Anhinga anhinga', 0.95 } } } }
	runImport(records, { { 'i1.CR3' } }, defaultPrefs())
	local before = #mock.state.photos[1]:keywordPaths()

	-- Re-run against a photo already carrying the previous result.
	local previous = mock.state.photos[1]._plugin
	mock.reset({ prefs = defaultPrefs(), confirmAnswer = 'ok' })
	mock.state.prefs.resultsPath = RESULTS
	local photo = mock.addPhoto('i1.CR3', {})
	for k, v in pairs(previous) do photo._plugin[k] = v end
	mock.install(PLUGIN)
	package.loaded['MelampusJson'] = nil; package.loaded['MelampusRules'] = nil
	package.loaded['MelampusLog'] = nil
	assert(loadfile(PLUGIN .. '/MelampusImport.lua'))()
	t.equals(#mock.state.photos[1]:keywordPaths(), 0,
		'a re-run re-applied keywords it had already written')
end)

os.remove(RESULTS)
return t.summary()
