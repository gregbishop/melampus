--[[
Runs the real plugin files against a mock Lightroom SDK: MelampusImport.lua end
to end, and MelampusAnalyze.lua and MelampusSettings.lua loaded fresh under the
mock, on a fake macOS and a fake Windows Lightroom.

This is the test that was missing. Everything before it checked decision logic
in isolation; this executes the actual files the plugin loads, with a real
results JSON on disk, and asserts on what landed in the catalog, on the command
the plugin builds for the executable beside it, and on what its dialogs say.

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

-- os.tmpname() creates the file; the suite writes it and removes it at the end.
local RESULTS = os.tmpname()

--- Drop the plugin's modules so the next load runs them fresh under the mock.
local function unloadPlugin()
	for _, name in ipairs({ 'MelampusJson', 'MelampusRules', 'MelampusLog', 'MelampusAnalyze' }) do
		package.loaded[name] = nil
	end
end

--- Load one plugin file fresh, dropping the modules first, and hand back what
--- it returns.
local function loadPluginFile(name)
	unloadPlugin()
	return dofile(PLUGIN .. '/' .. name .. '.lua')
end

--- Reset the mock, install it for a plugin folder (this one by default), and
--- load one plugin file fresh under it: the shape every load outside runImport
--- takes.
local function loadUnderMock(name, resetOptions, folder, installOptions)
	mock.reset(resetOptions)
	mock.install(folder or PLUGIN, installOptions)
	return loadPluginFile(name)
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
	local Rules = loadPluginFile('MelampusRules')
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
-- `prefs` overrides individual preferences on top of the defaults.
local function runAnalysis(existing, prefs)
	runImport({}, { { 'fresh_01.CR3' }, { 'fresh_02.CR3' } }, defaultPrefs(prefs),
		{ existing = existing })
	return mock.state.executed or {}
end

-- The executable beside this plugin, on the fake macOS Lightroom the suite
-- runs the import under; the mock reports it present when a test says so.
local MAC_EXECUTABLE = PLUGIN .. '/melampus'

-- A fake Windows Lightroom: the plugin in the per-user Modules folder, the
-- previews in the temp folder the mock names for WIN_ENV.
local WIN_PLUGIN = 'C:\\Users\\photographer\\AppData\\Roaming\\Adobe\\Lightroom\\Modules\\Melampus.lrplugin'
local WIN_EXECUTABLE = WIN_PLUGIN .. '\\melampus.exe'
local WIN_TEMP = 'C:\\Users\\photographer\\AppData\\Local\\Temp'
local WIN_PREVIEWS = WIN_TEMP .. '\\melampus-previews-1'

--- Load MelampusAnalyze.lua under a fake Windows Lightroom, with or without
--- melampus.exe beside the plugin.
local function loadAnalyzeOnWindows(executablePresent)
	local existing = {}
	if executablePresent then existing[WIN_EXECUTABLE] = true end
	return loadUnderMock('MelampusAnalyze', { existing = existing }, WIN_PLUGIN, { windows = true })
end

--- The engine's place on the command line: `--backend <engine>` after the
--- profile when one is set, quoted as `quote` quotes it, nothing when not
--- (the CLI decides).
local function engineOption(engine, quote)
	if engine == nil or engine == '' then return '' end
	return ' --backend ' .. quote(engine)
end

--- The one line the import runs on macOS: the executable beside the plugin
--- over the first batch of previews in the mock's temp directory, the
--- enriched results next to the previews, the CLI's own output kept in temp,
--- every path single-quoted for sh. The whole line, so nothing of a Python
--- checkout (python, .venv, tools/, cd) can be in it, wherever the clone is.
--- `engine` is the engine preference, on the line as --backend when set.
local function macCommand(engine)
	local temp = mock.state.tempDir
	local previews = temp .. '/melampus-previews-1'
	return string.format("'%s' '%s' --profile 'wildlife'%s --plugin-out '%s/results.json' --yes >'%s/melampus-cli.log' 2>&1",
		MAC_EXECUTABLE, previews, engineOption(engine, function(w) return "'" .. w .. "'" end),
		previews, temp)
end

--- The same line for a fake Windows Lightroom, as cmd.exe needs it: every
--- path double-quoted (single quotes mean nothing to cmd.exe), and the whole
--- line wrapped in a pair of its own: cmd.exe /c strips the first and last
--- quote of a line that starts with one and holds more than two, so the ones
--- around each path survive.
local function windowsCommand(engine)
	return string.format(
		'""%s" "%s" --profile "wildlife"%s --plugin-out "%s\\results.json" --yes >"%s\\melampus-cli.log" 2>&1"',
		WIN_EXECUTABLE, WIN_PREVIEWS, engineOption(engine, function(w) return '"' .. w .. '"' end),
		WIN_PREVIEWS, WIN_TEMP)
end

--- What Analyze.run says when the executable is not beside the plugin: the
--- file expected, named as a file (the folder's own name holds "melampus"),
--- and the folder that should hold it. No setup instruction; the user copies
--- one file.
local function missingExecutableMessage(executableName, folder)
	return 'Melampus could not find its analysis program.\n\n'
		.. 'The plugin folder should contain a file named ' .. executableName .. ':\n' .. folder
		.. '\n\nCopy it there from the Melampus download and try again.'
end

t.test('analysing runs the executable beside the plugin, in one command', function()
	local executed = runAnalysis({ [MAC_EXECUTABLE] = true })
	t.equals(#executed, 1, 'expected one command for identification and enrichment together')
	t.equals(executed[1], macCommand(), 'not the one command for the executable beside the plugin')
end)

t.test('every run exports its previews afresh', function()
	-- exportPreviews skips a preview that is already on disk. If runs share
	-- the machine's temp directory, the previews one run leaves are found by
	-- the next, the mock's requestJpegThumbnail is never called, and the
	-- outcome depends on what an earlier run (or an earlier suite) left behind.
	runAnalysis({ [MAC_EXECUTABLE] = true })
	t.equals(mock.state.previewsRequested, 2, 'the first run found previews it did not export')
	runAnalysis({ [MAC_EXECUTABLE] = true })
	t.equals(mock.state.previewsRequested, 2, "the second run found the first run's previews")
end)

t.test('with no results file configured, the executable analyses the selection', function()
	-- docs/plugin.md: leave the results path empty and the plugin analyses. A
	-- fresh install has no results file, so its first run must reach the offer
	-- and run the executable, not ask for a file from a Python checkout.
	runImport(nil, { { 'first_01.CR3' }, { 'first_02.CR3' } }, defaultPrefs(),
		{ existing = { [MAC_EXECUTABLE] = true } })
	t.isNotNil(dialogMatching('never been analysed'),
		'no offer to analyse; first dialog: ' .. tostring((mock.state.dialogs[1] or {}).body))
	t.isNil(dialogMatching('plugin_results.json'), 'asked for a results file instead of analysing')
	local executed = mock.state.executed or {}
	t.equals(#executed, 1, 'the executable did not run')
	t.equals(executed[1], macCommand(), 'not the one command for the executable beside the plugin')
end)

t.test('on Windows the command names melampus.exe with cmd.exe quoting', function()
	local Analyze = loadAnalyzeOnWindows(true)
	local ok, message = Analyze.run(WIN_PREVIEWS, WIN_PREVIEWS .. '\\results.json', 'wildlife')
	t.isTrue(ok, 'run failed: ' .. tostring(message))
	t.equals(mock.state.executed[1], windowsCommand(),
		'not the one command for melampus.exe beside the plugin, as cmd.exe needs it')
end)

t.test('the analyse offer is worded for both platforms', function()
	-- Windows is a covered invocation path; a Windows user is not on a Mac.
	runAnalysis({})
	local offer = dialogMatching('never been analysed')
	t.isNotNil(offer, 'no offer to analyse')
	t.isNil(string.find(offer, '%f[%a]Mac%f[%A]'), 'the offer says Mac:\n' .. offer)
end)

t.test('the "%" refusal names the path that has it, and the fix for that path', function()
	-- cmd.exe rewrites %NAME% even inside quotes; the plugin refuses rather
	-- than run against a path the user never named. Moving the plugin is the
	-- fix only when the "%" is in the plugin folder.
	local Analyze = loadAnalyzeOnWindows(true)
	local previews = 'C:\\Users\\photo%grapher\\AppData\\Local\\Temp\\melampus-previews-1'
	local ok, message = Analyze.run(previews, previews .. '\\results.json', 'wildlife')
	t.isFalse(ok, 'ran with a "%" in the previews path')
	t.isNotNil(string.find(message, previews, 1, true),
		'the message does not name the previews path:\n' .. message)
	t.isNil(string.find(message, 'Move the plugin', 1, true),
		'moving the plugin would not fix the previews path:\n' .. message)

	local results = 'D:\\out%put\\results.json'
	ok, message = Analyze.run(WIN_PREVIEWS, results, 'wildlife')
	t.isFalse(ok, 'ran with a "%" in the results path')
	t.isNotNil(string.find(message, results, 1, true),
		'the message does not name the results path:\n' .. message)
	t.isNil(mock.state.executed, 'ran a command through a path with "%"')
end)

-- ── the engine preference reaches the command line (card #403) ─────────────
-- Done-when 1: given a preference named engine with one of mlx, ollama,
-- openai, claude, when the plugin builds the CLI command, then the CLI
-- receives it. Done-when 2: given no preference, the plugin passes no
-- --backend and the CLI's default applies.
local ENGINES = { 'mlx', 'ollama', 'openai', 'claude' }

t.test('each engine preference reaches the command line as --backend', function()
	for _, engine in ipairs(ENGINES) do
		local executed = runAnalysis({ [MAC_EXECUTABLE] = true }, { engine = engine })
		t.equals(#executed, 1, engine .. ': expected one command')
		t.equals(executed[1], macCommand(engine), engine .. ' did not reach the command line as --backend')
	end
end)

t.test('no engine preference means no --backend on the command line', function()
	local executed = runAnalysis({ [MAC_EXECUTABLE] = true })
	t.equals(#executed, 1, 'expected one command')
	t.equals(executed[1], macCommand(), 'the plugin chose an engine the user never set')
end)

t.test('an unknown engine preference is refused before anything runs', function()
	local executed = runAnalysis({ [MAC_EXECUTABLE] = true }, { engine = 'anthropic' })
	t.equals(#executed, 0, 'ran a command with an engine the CLI would reject')
	local message = dialogMatching('anthropic')
	t.isNotNil(message, 'no dialog names the engine that was set')
	for _, engine in ipairs(ENGINES) do
		t.isNotNil(string.find(message, engine, 1, true), 'the dialog does not name ' .. engine)
	end
end)

t.test('on Windows the engine is double-quoted for cmd.exe', function()
	local Analyze = loadAnalyzeOnWindows(true)
	local ok, message = Analyze.run(WIN_PREVIEWS, WIN_PREVIEWS .. '\\results.json', 'wildlife', 'claude')
	t.isTrue(ok, 'run failed: ' .. tostring(message))
	t.equals(mock.state.executed[1], windowsCommand('claude'),
		'the engine is not double-quoted for cmd.exe')
end)

t.test('a missing executable names the plugin folder and the file it should hold', function()
	local executed = runAnalysis({})
	t.equals(#executed, 0, 'ran a command with no executable to run')
	local message = dialogMatching(PLUGIN)
	t.isNotNil(message, 'no dialog names the plugin folder ' .. PLUGIN)
	-- The import adds where it left the previews; the rest is Analyze.run's
	-- message, whole, so it cannot carry a setup instruction.
	t.equals(message, missingExecutableMessage('melampus', PLUGIN)
		.. '\n\nPreviews kept at:\n' .. mock.state.tempDir .. '/melampus-previews-1',
		'not the message for a missing executable')
end)

t.test('a missing executable on Windows names melampus.exe and the plugin folder', function()
	local Analyze = loadAnalyzeOnWindows(false)
	local ok, message = Analyze.run(WIN_PREVIEWS, WIN_PREVIEWS .. '\\results.json', 'wildlife')
	t.isFalse(ok, 'ran with no executable present')
	t.isNil(mock.state.executed, 'ran a command with no executable to run')
	t.equals(message, missingExecutableMessage('melampus.exe', WIN_PLUGIN),
		'not the message for a missing melampus.exe')
end)

-- ── the Settings dialog describes the plugin as it is now ──────────────────
-- The offer and Info.lua stopped saying "Mac" and stopped presenting a results
-- file worked out elsewhere as the plugin; the Settings dialog is read by the
-- same Windows user and must say the same thing.

--- Every string the dialog shows, from the view tree the mock kept: titles
--- (static text, group boxes, checkboxes, buttons) and tooltips, in order.
local function dialogStrings(view, out)
	out = out or {}
	if type(view) ~= 'table' then return out end
	for _, key in ipairs({ 'title', 'tooltip' }) do
		if type(view[key]) == 'string' then out[#out + 1] = view[key] end
	end
	for _, child in ipairs(view) do dialogStrings(child, out) end
	return out
end

--- The view bound to a preference, and the group box that holds it.
local function viewBoundTo(view, key, group)
	if type(view) ~= 'table' then return nil end
	if view.kind == 'group_box' then group = view end
	if view.value == key then return view, group end
	for _, child in ipairs(view) do
		local found, holder = viewBoundTo(child, key, group)
		if found then return found, holder end
	end
	return nil
end

t.test('the Settings dialog is worded for both platforms and the executable flow', function()
	loadUnderMock('MelampusSettings', { prefs = defaultPrefs() })
	local dialog = mock.state.dialogs[1]
	t.isNotNil(dialog and dialog.modal and dialog.contents or nil,
		'the Settings dialog was not presented with its contents')
	local strings = dialogStrings(dialog.contents)
	local text = table.concat(strings, '\n')
	for _, platform in ipairs({ 'Mac', 'Finder', 'Explorer' }) do
		t.isNil(string.find(text, '%f[%a]' .. platform .. '%f[%A]'),
			'the Settings dialog says ' .. platform .. ':\n' .. text)
	end

	-- The opening text says what the plugin does: it analyses, here.
	local intro = strings[1]
	t.isNotNil(string.find(intro, 'analys', 1, true),
		'the opening text does not say the plugin analyses the photos:\n' .. intro)

	-- The results file is the optional import of results produced elsewhere,
	-- not a required first step, and it is --plugin-out output that is read.
	local field, group = viewBoundTo(dialog.contents, 'resultsPath')
	t.isNotNil(field, 'no field is bound to resultsPath')
	t.isNotNil(group, 'the results-file field is not in a group box')
	t.isNotNil(string.find(string.lower(group.title), 'optional', 1, true),
		'the results-file group does not say it is optional:\n' .. group.title)
	t.isNotNil(string.find(tostring(field.tooltip), '--plugin-out', 1, true),
		'the results-file tooltip does not name --plugin-out:\n' .. tostring(field.tooltip))
	t.isNil(string.find(text, 'json-out', 1, true), 'the dialog still names --json-out:\n' .. text)
	-- With no first step, nothing is numbered as a step.
	for _, s in ipairs(strings) do
		t.isNil(string.match(s, '^Step %d'), 'numbered as a step when there is no first step: ' .. s)
	end
end)

os.remove(RESULTS)
mock.cleanUp()
return t.summary()
