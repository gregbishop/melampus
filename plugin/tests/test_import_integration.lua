--[[
Runs the real MelampusImport.lua against a mock Lightroom SDK.

This is the test that was missing. Everything before it checked decision logic
in isolation; this executes the actual file the plugin loads, with a real
results JSON on disk, and asserts on what landed in the catalog.

It is what would have caught the two failures seen in a real catalog: keywords
failing to attach, and a counter reporting success for photos that got nothing.
--]]

local t = require('harness')
local mock = require('lrmock')

--- Locate the plugin relative to this file, so the suite runs from a clone at any
--- path and from any working directory. MELAMPUS_PLUGIN overrides it.
local function pluginPath()
	local here = debug.getinfo(1, 'S').source:match('^@(.*)[/\\]') or '.'
	return here .. '/../Melampus.lrplugin'
end

local PLUGIN = os.getenv('MELAMPUS_PLUGIN') or pluginPath()

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

--- Drop the plugin's modules so the next load runs them fresh under the mock.
local function unloadPlugin()
	for _, name in ipairs({ 'MelampusJson', 'MelampusRules', 'MelampusLog', 'MelampusAnalyze' }) do
		package.loaded[name] = nil
	end
end

--- Run the real import file end to end and hand back the resulting state.
-- `records` is what the results file holds, or nil for no results file
-- configured at all, as on a fresh install. `photos` is a list of
-- { fileName, rawMetadata, pluginProperties }; `options` goes through to
-- mock.reset (confirmAnswer, existing, dropWrites), with the offer accepted
-- unless it says otherwise. Raises if the import does.
local function runImport(records, photos, prefs, options)
	options = options or {}
	options.prefs = prefs or {}
	options.confirmAnswer = options.confirmAnswer or 'ok'
	mock.reset(options)
	if records then
		writeResults(RESULTS, records)
		mock.state.prefs.resultsPath = RESULTS
	end
	for _, spec in ipairs(photos) do
		local photo = mock.addPhoto(spec[1], spec[2] or {})
		for k, v in pairs(spec[3] or {}) do photo._plugin[k] = v end
	end
	mock.install(PLUGIN)
	unloadPlugin()
	local ok, err = pcall(assert(loadfile(PLUGIN .. '/MelampusImport.lua')))
	if not ok then error('import raised: ' .. tostring(err), 2) end
	return true
end

local function defaultPrefs(extra)
	unloadPlugin()
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
	runImport(
		{ { file = 'd1.jpg', candidates = { { 'Snowy Egret', 'Egretta thula', 0.95 } } } },
		{ { 'd1.CR3' } }, defaultPrefs({ dryRun = true }), { confirmAnswer = 'cancel' })
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
	runImport(records, { { 'i1.CR3', {}, previous } }, defaultPrefs())
	t.equals(#mock.state.photos[1]:keywordPaths(), 0,
		'a re-run re-applied keywords it had already written')
end)

-- ── analysing photos that have never been seen ─────────────────────────────
t.test('unanalysed photos trigger an offer to analyse them', function()
	-- A failure seen in practice: selecting a folder the pipeline had never seen did
	-- nothing at all and reported success.
	runImport({}, { { 'never_seen_01.CR3' }, { 'never_seen_02.CR3' } }, defaultPrefs())
	local offered = false
	for _, d in ipairs(mock.state.dialogs) do
		if d.body and string.find(d.body, 'never been analysed', 1, true) then
			offered = true
		end
	end
	t.isTrue(offered, 'silently did nothing instead of offering to analyse')
end)

t.test('declining the offer leaves the photos untouched', function()
	runImport({}, { { 'untouched.CR3' } }, defaultPrefs(), { confirmAnswer = 'cancel' })
	t.equals(#mock.state.photos[1]:keywordPaths(), 0)
	t.isNil(mock.state.photos[1]:getRawMetadata('rating'))
end)

-- ── writes that report success but do not land ─────────────────────────────
-- The single worst failure this project has had: a run announcing 1301 photos
-- updated while the catalog held keywords for about 37. A counter next to a write
-- is not evidence the write happened, so applyPlans reads the catalog back.
--
-- `dropWrites` in the mock accepts a write and discards it silently, which is
-- exactly what that looked like from the plugin's side.

--- Run the import with a chosen number of writes swallowed for one photo.
local function runWithDroppedWrites(fileName, drops, records, photos)
	runImport(records, photos, defaultPrefs(), { dropWrites = { [fileName] = drops } })
end

local function logMatching(needle)
	for _, line in ipairs(mock.state.logLines) do
		if string.find(line, needle, 1, true) then return line end
	end
	return nil
end

local function dialogMatching(needle)
	for _, d in ipairs(mock.state.dialogs) do
		if d.body and string.find(d.body, needle, 1, true) then return d.body end
	end
	return nil
end

t.test('a write that silently fails is retried and lands', function()
	-- Two swallowed operations covers the rating and the first keyword of the
	-- first pass; the retry then writes them for real.
	runWithDroppedWrites('flaky.CR3', 2,
		{ { file = 'flaky.jpg', candidates = { { 'Osprey', 'Pandion haliaetus', 0.97 } } } },
		{ { 'flaky.CR3' } })

	t.isTrue(logMatching('did not land; retrying') ~= nil,
		'the failed write was never detected')
	local paths = mock.state.photos[1]:keywordPaths()
	t.isTrue(#paths > 0, 'the retry did not actually write the keyword')
end)

t.test('a write that fails twice is reported, not counted as success', function()
	-- A very large drop count means every attempt is swallowed, including the retry.
	runWithDroppedWrites('broken.CR3', 999,
		{ { file = 'broken.jpg', candidates = { { 'Osprey', 'Pandion haliaetus', 0.97 } } } },
		{ { 'broken.CR3' } })

	t.equals(#mock.state.photos[1]:keywordPaths(), 0, 'the mock did not drop the writes')
	t.isTrue(logMatching('failed twice') ~= nil, 'a permanent write failure went unreported')
	t.isTrue(dialogMatching('Changed 0 photos') ~= nil,
		'reported changing a photo whose writes never landed')
end)

t.test('a photo whose writes failed is not stamped as processed', function()
	-- Otherwise the failure is permanent: the next run treats it as already done
	-- and skips it forever.
	runWithDroppedWrites('broken.CR3', 999,
		{ { file = 'broken.jpg', candidates = { { 'Osprey', 'Pandion haliaetus', 0.97 } } } },
		{ { 'broken.CR3' } })

	t.isNil(mock.state.photos[1]:getPropertyForPlugin(nil, 'processedAt'),
		'stamped a photo as processed when nothing was written to it')
end)

t.test('one failing photo does not stop the others', function()
	runWithDroppedWrites('broken.CR3', 999, {
		{ file = 'broken.jpg', candidates = { { 'Osprey', 'Pandion haliaetus', 0.97 } } },
		{ file = 'fine.jpg', candidates = { { 'Snowy Egret', 'Egretta thula', 0.97 } } },
	}, { { 'broken.CR3' }, { 'fine.CR3' } })

	local byName = {}
	for _, photo in ipairs(mock.state.photos) do byName[photo.fileName] = photo end
	t.equals(#byName['broken.CR3']:keywordPaths(), 0)
	t.isTrue(#byName['fine.CR3']:keywordPaths() > 0,
		'a neighbouring failure took down a healthy write')
	t.isTrue(dialogMatching('Changed 1 photos') ~= nil,
		'the count did not exclude the photo that failed')
end)

-- ── analysing runs the executable beside the plugin (card #401) ────────────
-- The executable ships inside Melampus.lrplugin: `melampus` on macOS,
-- `melampus.exe` on Windows. The plugin runs it with --plugin-out, one command,
-- and never consults a Python environment.

--- Run the import with photos the results file has never seen, accept the
--- offer to analyse, and hand back the commands the shell was given.
local function runAnalysis(existing)
	runImport({}, { { 'fresh_01.CR3' }, { 'fresh_02.CR3' } }, defaultPrefs(),
		{ existing = existing })
	return mock.state.executed or {}
end

--- Load MelampusAnalyze.lua under the installed mock, fresh.
local function loadAnalyze()
	unloadPlugin()
	return dofile(PLUGIN .. '/MelampusAnalyze.lua')
end

local WIN_PLUGIN = 'C:\\Users\\photographer\\AppData\\Roaming\\Adobe\\Lightroom\\Modules\\Melampus.lrplugin'
local WIN_PREVIEWS = 'C:\\Users\\photographer\\AppData\\Local\\Temp\\melampus-previews-1'

--- The words a setup instruction would use; none belongs in a plugin dialog.
local function assertNoSetupInstructions(message)
	local lower = string.lower(message)
	for _, word in ipairs({ 'terminal', 'pip', 'venv', 'powershell', 'python' }) do
		t.isNil(string.find(lower, word, 1, true), 'the message tells the user about ' .. word)
	end
	t.isNil(string.find(lower, '%f[%a]uv%f[%A]'), 'the message tells the user about uv')
end

--- The command must name the executable and nothing of a Python checkout.
local function assertRunsTheExecutable(command, executable)
	t.isNotNil(string.find(command, executable, 1, true),
		'the command does not name ' .. executable .. ':\n' .. command)
	t.isNotNil(string.find(command, '--plugin-out', 1, true),
		'the command does not ask for the enriched results:\n' .. command)
	for _, forbidden in ipairs({ 'python', '.venv', 'tools/', 'tools\\', 'make_plugin_results', 'json-out' }) do
		t.isNil(string.find(command, forbidden, 1, true),
			'the command still goes through ' .. forbidden .. ':\n' .. command)
	end
end

t.test('analysing runs the executable beside the plugin, in one command', function()
	local executable = PLUGIN .. '/melampus'
	local executed = runAnalysis({ [executable] = true })
	t.equals(#executed, 1, 'expected one command for identification and enrichment together')
	assertRunsTheExecutable(executed[1], "'" .. executable .. "'")
	t.equals(string.sub(executed[1], 1, #executable + 2), "'" .. executable .. "'",
		'the command does not start with the executable:\n' .. executed[1])
	t.isNil(string.find(executed[1], 'cd ', 1, true),
		'the command changes directory, which only a checkout needed:\n' .. executed[1])
end)

t.test('with no results file configured, the executable analyses the selection', function()
	-- docs/plugin.md: leave the results path empty and the plugin analyses. A
	-- fresh install has no results file, so its first run must reach the offer
	-- and run the executable, not ask for a file from a Python checkout.
	local executable = PLUGIN .. '/melampus'
	runImport(nil, { { 'first_01.CR3' }, { 'first_02.CR3' } }, defaultPrefs(),
		{ existing = { [executable] = true } })
	t.isNotNil(dialogMatching('never been analysed'),
		'no offer to analyse; first dialog: ' .. tostring((mock.state.dialogs[1] or {}).body))
	t.isNil(dialogMatching('plugin_results.json'), 'asked for a results file instead of analysing')
	local executed = mock.state.executed or {}
	t.equals(#executed, 1, 'the executable did not run')
	assertRunsTheExecutable(executed[1], "'" .. executable .. "'")
end)

t.test('on Windows the command names melampus.exe with cmd.exe quoting', function()
	mock.reset({ existing = { [WIN_PLUGIN .. '\\melampus.exe'] = true } })
	mock.install(WIN_PLUGIN, { windows = true })
	local Analyze = loadAnalyze()
	local ok, message = Analyze.run(WIN_PREVIEWS, WIN_PREVIEWS .. '\\results.json', 'wildlife')
	t.isTrue(ok, 'run failed: ' .. tostring(message))
	local command = mock.state.executed[1]
	assertRunsTheExecutable(command, '"' .. WIN_PLUGIN .. '\\melampus.exe"')
	t.isNotNil(string.find(command, '"' .. WIN_PREVIEWS .. '\\results.json"', 1, true),
		'the results path is not double-quoted for cmd.exe:\n' .. command)
	t.isNil(string.find(command, "'", 1, true), 'single quotes mean nothing to cmd.exe:\n' .. command)
	-- cmd.exe /c strips the first and last quote of a line that starts with one
	-- and holds more than two; the whole line is wrapped so the ones that
	-- matter survive.
	t.equals(string.sub(command, 1, 2), '""', 'the command is not wrapped for cmd.exe:\n' .. command)
	t.equals(string.sub(command, -1), '"', 'the command is not wrapped for cmd.exe:\n' .. command)
	t.isNil(string.find(command, 'cd ', 1, true),
		'the command changes directory, which only a checkout needed:\n' .. command)
end)

t.test('a missing executable names the plugin folder and the file it should hold', function()
	local executed = runAnalysis({})
	t.equals(#executed, 0, 'ran a command with no executable to run')
	local message = dialogMatching(PLUGIN)
	t.isNotNil(message, 'no dialog names the plugin folder ' .. PLUGIN)
	t.isNotNil(string.find(message, 'melampus', 1, true), 'the dialog does not say what file is expected')
	assertNoSetupInstructions(message)
end)

t.test('a missing executable on Windows names melampus.exe and the plugin folder', function()
	mock.reset()
	mock.install(WIN_PLUGIN, { windows = true })
	local Analyze = loadAnalyze()
	local ok, message = Analyze.run(WIN_PREVIEWS, WIN_PREVIEWS .. '\\results.json', 'wildlife')
	t.isFalse(ok, 'ran with no executable present')
	t.isNil(mock.state.executed, 'ran a command with no executable to run')
	t.isNotNil(string.find(message, WIN_PLUGIN, 1, true), 'the message does not name the plugin folder:\n' .. message)
	t.isNotNil(string.find(message, 'melampus.exe', 1, true), 'the message does not name melampus.exe:\n' .. message)
	assertNoSetupInstructions(message)
end)

os.remove(RESULTS)
return t.summary()
