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

local PLUGIN = mock.PLUGIN

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

local loadPluginFile, loadUnderMock = mock.loadPluginFile, mock.loadUnderMock

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
	mock.unloadPlugin()
	local ok, err = pcall(assert(loadfile(PLUGIN .. '/MelampusImport.lua')))
	if not ok then error('import raised: ' .. tostring(err), 2) end
	return true
end

--- The defaults with dry run off, so the writes this suite asserts on land,
--- and `extra` over that.
local function defaultPrefs(extra)
	return mock.defaultPrefs({ dryRun = false }, extra)
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
local MAC_EXECUTABLE = mock.EXECUTABLE

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
--- every path single-quoted for sh, an apostrophe in it (a checkout or a
--- TMPDIR under a name like O'Brien) closed, escaped and reopened, as sh
--- needs it. The whole line, so nothing of a Python checkout (python, .venv,
--- tools/, cd) can be in it, wherever the clone is. `engine` is the engine
--- preference, on the line as --backend when set. `variable` and `key` are
--- the cloud engine's key variable and the key LrPasswords holds for it: set
--- ahead of the executable in its environment (`VAR='key' command`, as sh
--- sets one) when given, nothing when not.
local function macCommand(engine, variable, key)
	local temp = mock.state.tempDir
	local previews = temp .. '/melampus-previews-1'
	local environment = variable and (variable .. '=' .. mock.sh(key) .. ' ') or ''
	return environment .. table.concat({
		mock.sh(MAC_EXECUTABLE), mock.sh(previews),
		'--profile', mock.sh('wildlife') .. engineOption(engine, mock.sh),
		'--plugin-out', mock.sh(previews .. '/results.json'),
		'--yes',
		'>' .. mock.sh(temp .. '/melampus-cli.log') .. ' 2>&1',
	}, ' ')
end

--- The same line for a fake Windows Lightroom, as cmd.exe needs it: every
--- path double-quoted (single quotes mean nothing to cmd.exe), the key set
--- with `set "VAR=key" &&` ahead of the executable when given, and the whole
--- line wrapped in a pair of quotes of its own: cmd.exe /c strips the first
--- and last quote of a line that starts with one and holds more than two, so
--- the ones around each path survive.
local function windowsCommand(engine, variable, key)
	local environment = variable and ('set "' .. variable .. '=' .. key .. '" && ') or ''
	return string.format(
		'"%s"%s" "%s" --profile "wildlife"%s --plugin-out "%s\\results.json" --yes >"%s\\melampus-cli.log" 2>&1"',
		environment, WIN_EXECUTABLE, WIN_PREVIEWS, engineOption(engine, function(w) return '"' .. w .. '"' end),
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
-- --backend and the CLI's default applies. Card #423 adds the two
-- subscription CLIs, claude-code and codex, which reach it the same way.
local Rules = loadPluginFile('MelampusRules')
local ENGINES = Rules.ENGINES

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
	-- Absent by name, not by omission: the readme's install step puts the
	-- real executable beside this plugin, and a mock that then looked at the
	-- disk would find it and run it, so the test must hold on an installed
	-- checkout as on a bare one.
	local executed = runAnalysis({ [MAC_EXECUTABLE] = false })
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
-- same Windows user and must say the same thing. The dialog's strings and
-- bound fields are read through the mock's walk of the recorded view tree.

t.test('the Settings dialog is worded for both platforms and the executable flow', function()
	loadUnderMock('MelampusSettings', { prefs = defaultPrefs() })
	local dialog = mock.state.dialogs[1]
	t.isNotNil(dialog and dialog.modal and dialog.contents or nil,
		'the Settings dialog was not presented with its contents')
	local strings = mock.dialogStrings(dialog.contents)
	local text = table.concat(strings, '\n')
	for _, platform in ipairs({ 'Mac', 'Finder', 'Explorer' }) do
		t.isNil(string.find(text, '%f[%a]' .. platform .. '%f[%A]'),
			'the Settings dialog says ' .. platform .. ':\n' .. text)
	end
	-- Windows has no keychain: the key's tooltip is read there too.
	t.isNil(string.find(string.lower(text), 'keychain', 1, true),
		'the Settings dialog says keychain, which Windows has not:\n' .. text)

	-- The opening text says what the plugin does: it analyses, here.
	local intro = strings[1]
	t.isNotNil(string.find(intro, 'analys', 1, true),
		'the opening text does not say the plugin analyses the photos:\n' .. intro)

	-- The results file is the optional import of results produced elsewhere,
	-- not a required first step, and it is --plugin-out output that is read.
	local field, group = mock.viewBoundTo(dialog.contents, 'resultsPath')
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

-- ── the picked cloud engine's key reaches the executable (card #405) ──────
-- The key lives in LrPasswords. When the picked engine needs one, the
-- command carries it in the executable's environment, in the variable the
-- executable reads (MELAMPUS_OPENAI_KEY, MELAMPUS_ANTHROPIC_KEY), never as
-- an argument, never in the log, and never for an engine that needs none.
local KEY = 'stored-in-lrpasswords-not-a-real-key-4c1e'

--- Run the import for `engine` with the given LrPasswords contents, on macOS,
--- and hand back the one command the shell was given.
local function commandWithKeys(engine, passwords)
	runImport({}, { { 'fresh_01.CR3' }, { 'fresh_02.CR3' } }, defaultPrefs({ engine = engine }),
		{ existing = { [MAC_EXECUTABLE] = true }, passwords = passwords })
	local executed = mock.state.executed or {}
	t.equals(#executed, 1, 'expected one command')
	return executed[1]
end

t.test('the stored key for the picked cloud engine goes to the executable in its environment', function()
	-- The whole line: the variable set once, ahead of the executable, never
	-- as an argument, and not the other engine's variable.
	local command = commandWithKeys('openai', { MELAMPUS_OPENAI_KEY = KEY })
	t.equals(command, macCommand('openai', 'MELAMPUS_OPENAI_KEY', KEY),
		'not the command with the OpenAI key set in the environment')

	command = commandWithKeys('claude', { MELAMPUS_ANTHROPIC_KEY = KEY, MELAMPUS_OPENAI_KEY = 'other' })
	t.equals(command, macCommand('claude', 'MELAMPUS_ANTHROPIC_KEY', KEY),
		'not the command with the Claude key alone set in the environment')
end)

t.test('the key is never logged', function()
	commandWithKeys('openai', { MELAMPUS_OPENAI_KEY = KEY })
	t.isTrue(#mock.state.logLines > 0, 'the run was not logged at all')
	for _, line in ipairs(mock.state.logLines) do
		t.isNil(string.find(line, KEY, 1, true), 'the key was logged: ' .. line)
	end
end)

t.test('a local engine, a subscription CLI, or no engine, carries no key even when keys are stored', function()
	local stored = { MELAMPUS_OPENAI_KEY = KEY, MELAMPUS_ANTHROPIC_KEY = KEY }
	-- Card #423: the CLI engines bill to a subscription, never to a key
	-- here, so their line is the executable and --backend, nothing ahead.
	-- Which engines need no key is the plugin's own rule, Rules.keyVariable,
	-- pinned by name in test_rules.lua; walked over ENGINES here, so the
	-- next keyless engine is checked without a line here.
	for _, engine in ipairs(ENGINES) do
		if not Rules.keyVariable(engine) then
			local command = commandWithKeys(engine, stored)
			t.equals(command, macCommand(engine), engine .. ': a key travels with a run that needs none')
		end
	end
	t.equals(commandWithKeys('', stored), macCommand(''), 'a key travels with a run that names no engine')
end)

t.test('a cloud engine with no stored key runs without one, so the executable says what is missing', function()
	local command = commandWithKeys('openai', {})
	t.equals(command, macCommand('openai'), 'an empty key was set')
end)

t.test('on Windows the key is set for cmd.exe before the executable, once', function()
	local Analyze = loadUnderMock('MelampusAnalyze',
		{ existing = { [WIN_EXECUTABLE] = true }, passwords = { MELAMPUS_ANTHROPIC_KEY = KEY } },
		WIN_PLUGIN, { windows = true })
	local ok, message = Analyze.run(WIN_PREVIEWS, WIN_PREVIEWS .. '\\results.json', 'wildlife', 'claude')
	t.isTrue(ok, 'run failed: ' .. tostring(message))
	t.equals(mock.state.executed[1], windowsCommand('claude', 'MELAMPUS_ANTHROPIC_KEY', KEY),
		'not the command with the Claude key set for cmd.exe ahead of the executable')
	for _, line in ipairs(mock.state.logLines) do
		t.isNil(string.find(line, KEY, 1, true), 'the key was logged: ' .. line)
	end
end)

t.test('on Windows a stored key holding a character cmd.exe rewrites is refused before anything runs', function()
	-- Inside `set "VAR=value"` a double quote ends the quoted text and what
	-- follows is command text to cmd.exe, and %NAME% is expanded even inside
	-- quotes: the same rewriting the paths are refused for. A line feed ends
	-- the line itself, so what follows it is not the line the plugin built,
	-- and a carriage return is dropped. There is no way to escape any of
	-- them on a cmd.exe command line, so the key is refused, the way to fix
	-- it named, and the key itself shown nowhere.
	for _, key in ipairs({
		'sk-not-a-real-key" & calc & "', 'sk-not-a-real-key-%TEMP%',
		'sk-not-a-real-key\ncalc', 'sk-not-a-real-key\r',
	}) do
		local Analyze = loadUnderMock('MelampusAnalyze',
			{ existing = { [WIN_EXECUTABLE] = true }, passwords = { MELAMPUS_OPENAI_KEY = key } },
			WIN_PLUGIN, { windows = true })
		local ok, message = Analyze.run(WIN_PREVIEWS, WIN_PREVIEWS .. '\\results.json', 'wildlife', 'openai')
		t.isFalse(ok, 'ran with a key cmd.exe would rewrite: ' .. key)
		t.isNil(mock.state.executed, 'a command carrying the key reached cmd.exe: ' .. key)
		t.isNotNil(string.find(message, 'Settings', 1, true),
			'the message does not say where to enter the key again:\n' .. tostring(message))
		t.isNil(string.find(message, key, 1, true), 'the message shows the key:\n' .. message)
		for _, line in ipairs(mock.state.logLines) do
			t.isNil(string.find(line, key, 1, true), 'the key was logged: ' .. line)
		end
	end
end)

t.test('on macOS a key holding shell characters travels intact, single-quoted for sh', function()
	-- sh gets the key through quote(): an apostrophe closed, escaped and
	-- reopened; a double quote, a percent sign and a dollar mean nothing
	-- inside single quotes. So nothing is refused there.
	local key = "sk-not-a-real-key-o'brien\"%TEMP%$HOME"
	local command = commandWithKeys('openai', { MELAMPUS_OPENAI_KEY = key })
	t.equals(command, macCommand('openai', 'MELAMPUS_OPENAI_KEY', key),
		'not the command with the key single-quoted for sh')
end)

-- ── asking the executable which engines can run here (card #405) ───────────
-- The dialog's picker shows what `melampus --detect-engines` says. The plugin
-- runs the executable beside it once, reads the JSON it printed, and hands the
-- decoded list to Rules.engineItems; a missing executable is the same message
-- the analysis gives, never a crash.
--- Load MelampusAnalyze.lua under a fake macOS Lightroom with the executable
--- beside the plugin, played by mock.answersDetection(text, code).
local function loadAnalyzeAnswering(text, code)
	return loadUnderMock('MelampusAnalyze',
		{ existing = { [MAC_EXECUTABLE] = true }, onExecute = mock.answersDetection(text, code) })
end

--- The one line detection runs on macOS: the executable beside the plugin
--- asked for its verdicts, its stdout to a file in the mock's temp directory
--- and its stderr to the CLI log there, every path single-quoted for sh.
local function macDetectionCommand()
	local temp = mock.state.tempDir
	return string.format("'%s' --detect-engines >'%s/melampus-engines.json' 2>'%s/melampus-cli.log'",
		MAC_EXECUTABLE, temp, temp)
end

t.test('detection runs the executable once with --detect-engines and returns the decoded list', function()
	local Analyze = loadAnalyzeAnswering(mock.detectionText())
	local verdicts, problem = Analyze.detectEngines()
	t.isNotNil(verdicts, 'no verdicts: ' .. tostring(problem))
	t.equals(#mock.state.executed, 1, 'detection should run the executable exactly once')
	t.equals(mock.state.executed[1], macDetectionCommand(), 'not the one command that asks for the verdicts')
	t.equals(#verdicts, #ENGINES, 'one verdict per engine the plugin knows')
	t.equals(verdicts[2].engine, 'ollama')
	t.isFalse(verdicts[2].available)
	t.isNotNil(string.find(verdicts[2].reason, mock.OLLAMA_INSTALL, 1, true))
end)

t.test('a missing executable makes detection say so, with the plugin folder and the file', function()
	-- Absent by name, as the analysis's sibling above: the mock must not look
	-- at the disk, where the readme's install step may have put the real one.
	local Analyze = loadUnderMock('MelampusAnalyze', { existing = { [MAC_EXECUTABLE] = false } })
	local verdicts, problem = Analyze.detectEngines()
	t.isNil(verdicts)
	t.isNil(mock.state.executed, 'ran a command with no executable to run')
	t.equals(problem, missingExecutableMessage('melampus', PLUGIN),
		'not the message for a missing executable')
end)

t.test('an executable that fails or prints no list makes detection say so, never raise', function()
	local verdicts, problem = loadAnalyzeAnswering('Traceback (most recent call last)', 1).detectEngines()
	t.isNil(verdicts, 'a failed run produced verdicts')
	t.isNotNil(string.find(problem, 'exit 1', 1, true), 'the message does not give the exit code:\n' .. tostring(problem))

	-- Exit 0 with something other than the list: what the executable printed
	-- went to the engines file, its stderr to the CLI log. The message names
	-- both, each as what it is; neither is "the log" on its own. The temp
	-- directory is the one the mock made for this load.
	local function namesBothFiles(message)
		local temp = mock.state.tempDir
		t.isNotNil(string.find(message, 'What it printed is in:\n' .. temp .. '/melampus-engines.json', 1, true),
			'the message does not say what the engines file is and where:\n' .. tostring(message))
		t.isNotNil(string.find(message, temp .. '/melampus-cli.log', 1, true),
			'the message does not name the CLI log:\n' .. tostring(message))
		t.isNil(string.find(message, 'See the log:', 1, true),
			'the message calls the engines file the log:\n' .. tostring(message))
	end

	verdicts, problem = loadAnalyzeAnswering('{"not": "a list"}').detectEngines()
	t.isNil(verdicts, 'an object is not the verdict list')
	namesBothFiles(problem)

	verdicts, problem = loadAnalyzeAnswering('').detectEngines()
	t.isNil(verdicts, 'empty output is not the verdict list')
	namesBothFiles(problem)
end)

t.test('on Windows detection names melampus.exe with cmd.exe quoting', function()
	local Analyze = loadAnalyzeOnWindows(true)
	Analyze.detectEngines()
	-- Every path double-quoted and the whole line wrapped, as cmd.exe needs it.
	t.equals(mock.state.executed[1], string.format(
		'""%s" --detect-engines >"%s\\melampus-engines.json" 2>"%s\\melampus-cli.log""',
		WIN_EXECUTABLE, WIN_TEMP, WIN_TEMP),
		'not the one command that asks melampus.exe for the verdicts, as cmd.exe needs it')
end)

t.test('on Windows detection refuses a path with "%" the way the run does, naming the fix', function()
	-- Detection redirects to the same temp-folder paths the run refuses when
	-- they hold "%", and its executable is in the same plugin folder. Run
	-- through cmd.exe unchecked, a rewritten path reports "could not ask"
	-- or "did not understand" instead of the refusal that names the fix.
	local folder = 'C:\\Users\\photo%grapher\\AppData\\Roaming\\Adobe\\Lightroom\\Modules\\Melampus.lrplugin'
	local Analyze = loadUnderMock('MelampusAnalyze',
		{ existing = { [folder .. '\\melampus.exe'] = true } }, folder, { windows = true })
	local verdicts, problem = Analyze.detectEngines()
	t.isNil(verdicts, 'detected through a plugin folder with "%"')
	t.isNil(mock.state.executed, 'ran detection through a plugin folder with "%"')
	t.isNotNil(string.find(problem, folder, 1, true),
		'the message does not name the plugin folder:\n' .. tostring(problem))
	t.isNotNil(string.find(problem, 'Move the plugin', 1, true),
		'the message does not name the fix for the plugin folder:\n' .. problem)

	local temp = 'C:\\Users\\photo%grapher\\AppData\\Local\\Temp'
	Analyze = loadUnderMock('MelampusAnalyze',
		{ existing = { [WIN_EXECUTABLE] = true }, windowsTemp = temp }, WIN_PLUGIN, { windows = true })
	verdicts, problem = Analyze.detectEngines()
	t.isNil(verdicts, 'detected through a temp folder with "%"')
	t.isNil(mock.state.executed, 'ran detection through a temp folder with "%"')
	t.isNotNil(string.find(problem, temp .. '\\melampus-engines.json', 1, true),
		'the message does not name the file in the temp folder:\n' .. tostring(problem))
	t.isNotNil(string.find(problem, 'Set TEMP', 1, true),
		'the message does not name the fix for the temp folder:\n' .. problem)
	t.isNil(string.find(problem, 'Move the plugin', 1, true),
		'moving the plugin would not fix the temp folder:\n' .. problem)
end)

os.remove(RESULTS)
mock.cleanUp()
return t.summary()
