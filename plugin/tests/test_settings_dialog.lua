--[[
Runs the real MelampusSettings.lua against the mock Lightroom SDK (card #405).

The engine picker shows what `melampus --detect-engines` says: the four
engines in the owner's order, the ones that cannot run here greyed with their
reason, a link to install Ollama when it is missing, and a password field for
the API key of the picked cloud engine, stored through LrPasswords and never
in the preferences, a file, or the log.
--]]

local t = require('harness')
local mock = require('lrmock')

local function pluginPath()
	local here = debug.getinfo(1, 'S').source:match('^@(.*)[/\\]') or '.'
	return here .. '/../Melampus.lrplugin'
end

local PLUGIN = os.getenv('MELAMPUS_PLUGIN') or pluginPath()
local EXECUTABLE = PLUGIN .. '/melampus'
local ENGINES = { 'mlx', 'ollama', 'openai', 'claude' }
local OLLAMA_DOWNLOAD = 'https://ollama.com/download'

local function verdict(engine, available, reason)
	return string.format('{"engine": %q, "available": %s, "reason": %q}', engine, tostring(available), reason)
end

--- The executable's --detect-engines answer on a Mac with no Ollama running.
local function detection(overrides)
	local reasons = {
		mlx = { true, 'runs locally on this Apple Silicon Mac' },
		ollama = { false, 'no Ollama server at http://127.0.0.1:11434; install it from ' .. OLLAMA_DOWNLOAD },
		openai = { true, 'API key required: set MELAMPUS_OPENAI_KEY (or OPENAI_API_KEY)' },
		claude = { true, 'API key required: set MELAMPUS_ANTHROPIC_KEY (or ANTHROPIC_API_KEY)' },
	}
	for engine, value in pairs(overrides or {}) do reasons[engine] = value end
	local parts = {}
	for _, engine in ipairs(ENGINES) do
		parts[#parts + 1] = verdict(engine, reasons[engine][1], reasons[engine][2])
	end
	return '[' .. table.concat(parts, ', ') .. ']'
end

local function defaultPrefs(extra)
	package.loaded['MelampusRules'] = nil
	local prefs = dofile(PLUGIN .. '/MelampusRules.lua').defaultSettings()
	for k, v in pairs(extra or {}) do prefs[k] = v end
	return prefs
end

local REPO = 'mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit'
local CANCEL_PATH = (os.getenv('TMPDIR') or '/tmp') .. '/melampus-data/cache/download-cancel'

--- The executable's --model-status answer: absent with the size by default.
local function modelStatus(overrides)
	local status = {
		repo = '"' .. REPO .. '"', installed = 'false', bytes_total = '18300000000',
		bytes_done = '0', path = 'null', cancel_path = '"' .. CANCEL_PATH .. '"',
	}
	for key, value in pairs(overrides or {}) do status[key] = value end
	local parts = {}
	for _, key in ipairs({ 'repo', 'installed', 'bytes_total', 'bytes_done', 'path', 'cancel_path' }) do
		parts[#parts + 1] = '"' .. key .. '": ' .. status[key]
	end
	return '{' .. table.concat(parts, ', ') .. '}'
end

local function writeFile(path, text, mode)
	local handle = assert(io.open(path, mode or 'w'))
	handle:write(text)
	handle:close()
end

--- The dialogs the mock recorded, the modal ones (the Settings dialog, with
--- its view tree) when `modal` is true, else the messages shown over it.
local function dialogsShown(modal)
	local found = {}
	for _, dialog in ipairs(mock.state.dialogs) do
		if (dialog.modal or false) == modal then found[#found + 1] = dialog end
	end
	return found
end

--- Open the real Settings dialog under the mock. `options.detection` is what
--- the executable prints for --detect-engines (nil: no executable beside the
--- plugin); `options.status` what it prints for --model-status (default: the
--- model absent); `options.download` plays --download-model: its `lines` land
--- in the progress file one per tick, `stderr` in the log, and it exits
--- `code`; `options.removeCode` is --remove-model's exit code; `options
--- .onDialog` plays the user while the dialog is up.
local function openSettings(options)
	options = options or {}
	mock.reset({
		prefs = defaultPrefs(options.prefs),
		existing = options.detection and { [EXECUTABLE] = true } or {},
		passwords = options.passwords,
		onExecute = function(command)
			local target = string.match(command, ">'([^']+)'")
			if string.find(command, '--detect-engines', 1, true) and target and options.detection then
				writeFile(target, options.detection)
			elseif string.find(command, '--model-status', 1, true) and target then
				writeFile(target, options.status or modelStatus())
			elseif string.find(command, '--download-model', 1, true) and target then
				local download = options.download or { lines = {}, code = 0 }
				local log = string.match(command, "2>'([^']+)'")
				for _, line in ipairs(download.lines) do
					mock.yield()
					writeFile(target, line .. '\n', 'a')
				end
				if download.stderr then writeFile(log, download.stderr, 'a') end
				return download.code
			elseif string.find(command, '--remove-model', 1, true) then
				return options.removeCode or 0
			end
			return 0
		end,
		onModalDialog = options.onDialog,
	})
	mock.install(PLUGIN)
	for _, name in ipairs({ 'MelampusJson', 'MelampusRules', 'MelampusLog', 'MelampusAnalyze' }) do
		package.loaded[name] = nil
	end
	local ok, err = pcall(assert(loadfile(PLUGIN .. '/MelampusSettings.lua')))
	t.isTrue(ok, 'the settings file raised: ' .. tostring(err))
	local modal = dialogsShown(true)
	t.equals(#modal, 1, 'expected exactly one dialog')
	return modal[1].contents
end

--- Every view of one kind in the tree, in order, each with its parent.
local function viewsOfKind(root, kind)
	local found = {}
	local function walk(node, parent)
		if type(node) ~= 'table' then return end
		if node.kind == kind then found[#found + 1] = { view = node, parent = parent } end
		for _, child in ipairs(node) do walk(child, node) end
	end
	walk(root, nil)
	return found
end

local function bindingKey(binding)
	if type(binding) == 'table' then return binding.key end
	return binding
end

--- The one popup_menu bound to prefs.engine.
local function enginePicker(contents)
	for _, entry in ipairs(viewsOfKind(contents, 'popup_menu')) do
		if bindingKey(entry.view.value) == 'engine' then return entry.view end
	end
	return nil
end

local function titlesMatching(contents, needle)
	local out = {}
	for _, entry in ipairs(viewsOfKind(contents, 'static_text')) do
		local title = entry.view.title
		if type(title) == 'string' and string.find(title, needle, 1, true) then out[#out + 1] = entry.view end
	end
	return out
end

--- The row of the engine group that holds the model's buttons: the one bound
--- to a property table with a `phase`. nil when there is none.
local function modelRow(contents)
	for _, kind in ipairs({ 'row', 'column' }) do
		for _, entry in ipairs(viewsOfKind(contents, kind)) do
			local bound = entry.view.bind_to_object
			if type(bound) == 'table' and bound.phase ~= nil then return entry.view, bound end
		end
	end
	return nil
end

-- ── the picker ─────────────────────────────────────────────────────────────
t.test('the settings dialog opens with a picker bound to prefs.engine listing the four engines in order', function()
	local contents = openSettings({ detection = detection() })
	local picker = enginePicker(contents)
	t.isNotNil(picker, 'no popup_menu bound to engine')
	local values = {}
	for _, item in ipairs(picker.items) do values[#values + 1] = item.value end
	t.equals(values[1], '', 'the first item leaves the choice to Melampus, the unset preference')
	for i, engine in ipairs(ENGINES) do
		t.equals(values[i + 1], engine, 'item ' .. (i + 1) .. ' of the picker')
	end
	t.equals(#values, 5)
end)

local function commandsRun(flag)
	local count = 0
	for _, command in ipairs(mock.state.executed or {}) do
		if string.find(command, flag, 1, true) then count = count + 1 end
	end
	return count
end

t.test('detection and the model status each run once, when the dialog opens', function()
	openSettings({ detection = detection() })
	t.equals(commandsRun('--detect-engines'), 1, 'the executable should be asked once about the engines')
	t.equals(commandsRun('--model-status'), 1, 'the executable should be asked once about the model')
	t.equals(#mock.state.executed, 2, 'nothing but detection and the status should run')
end)

t.test('where mlx cannot run the model is not asked about and there is no download row', function()
	local contents = openSettings({ detection = detection({ mlx = { false, 'needs Apple Silicon' } }) })
	t.equals(commandsRun('--model-status'), 0, 'the model was asked about where it cannot run')
	t.equals(#mock.state.executed, 1)
	t.isNil(modelRow(contents), 'a download row with no mlx to use it')
end)

t.test('engines that cannot run here are greyed and their reasons shown', function()
	local contents = openSettings({ detection = detection({ mlx = { false, 'needs Apple Silicon' } }) })
	local enabled = {}
	for _, item in ipairs(enginePicker(contents).items) do enabled[item.value] = item.enabled end
	t.isFalse(enabled.mlx, 'mlx should be greyed')
	t.isFalse(enabled.ollama, 'ollama should be greyed')
	t.isTrue(enabled.openai, 'openai should be available')
	t.isTrue(enabled.claude, 'claude should be available')
	t.isTrue(enabled[''], 'letting Melampus choose is always available')
	t.isTrue(#titlesMatching(contents, 'needs Apple Silicon') > 0, 'the mlx reason is not shown')
	t.isTrue(#titlesMatching(contents, 'no Ollama server') > 0, 'the ollama reason is not shown')
	t.equals(#titlesMatching(contents, 'API key required'), 0, 'an available engine needs no reason')
end)

t.test('with every engine available nothing is greyed', function()
	local contents = openSettings({ detection = detection({ ollama = { true, 'Ollama is answering at http://127.0.0.1:11434' } }) })
	for _, item in ipairs(enginePicker(contents).items) do
		t.isTrue(item.enabled, item.value .. ' was greyed')
	end
	t.equals(#titlesMatching(contents, 'Ollama'), 0, 'a reason was shown for an available engine')
end)

-- ── the Ollama link ────────────────────────────────────────────────────────
t.test('when ollama is unavailable a link opens the Ollama download page', function()
	local contents = openSettings({ detection = detection() })
	local links = titlesMatching(contents, OLLAMA_DOWNLOAD)
	local clickable = {}
	for _, view in ipairs(links) do
		if type(view.mouse_down) == 'function' then clickable[#clickable + 1] = view end
	end
	t.equals(#clickable, 1, 'expected exactly one clickable link to ' .. OLLAMA_DOWNLOAD)
	clickable[1].mouse_down()
	t.equals(#mock.state.openedUrls, 1, 'the click did not open the browser')
	t.equals(mock.state.openedUrls[1], OLLAMA_DOWNLOAD)
end)

t.test('when ollama is available there is no link', function()
	local contents = openSettings({ detection = detection({ ollama = { true, 'Ollama is answering at http://127.0.0.1:11434' } }) })
	for _, entry in ipairs(viewsOfKind(contents, 'static_text')) do
		t.isNil(entry.view.mouse_down, 'a clickable link is shown with nothing to install: ' .. tostring(entry.view.title))
	end
	t.equals(#titlesMatching(contents, OLLAMA_DOWNLOAD), 0)
end)

-- ── the API key ────────────────────────────────────────────────────────────
--- The password fields, keyed by the variable each is bound to.
local function keyFields(contents)
	local fields = {}
	for _, entry in ipairs(viewsOfKind(contents, 'password_field')) do
		fields[bindingKey(entry.view.value)] = entry
	end
	return fields
end

--- Whether a view (or the row holding it) is visible with `engine` picked,
--- through its visible binding's transform.
local function visibleFor(entry, engine)
	local binding = entry.view.visible or (entry.parent and entry.parent.visible)
	t.isNotNil(binding, 'the key field has no visible binding')
	t.equals(bindingKey(binding), 'engine', 'the key field is not shown by the engine')
	t.equals(type(binding.transform), 'function', 'the visible binding has no transform')
	return binding.transform(engine, mock.state.prefs)
end

t.test('a password field takes the key for openai and for claude, shown only when that engine is picked', function()
	local contents = openSettings({ detection = detection() })
	local fields = keyFields(contents)
	t.isNotNil(fields.MELAMPUS_OPENAI_KEY, 'no password field for the OpenAI key')
	t.isNotNil(fields.MELAMPUS_ANTHROPIC_KEY, 'no password field for the Claude key')
	local count = 0
	for _ in pairs(fields) do count = count + 1 end
	t.equals(count, 2, 'expected exactly two password fields')
	for _, engine in ipairs({ '', 'mlx', 'ollama', 'claude' }) do
		t.isFalse(visibleFor(fields.MELAMPUS_OPENAI_KEY, engine), 'the OpenAI key field shows for ' .. engine)
	end
	t.isTrue(visibleFor(fields.MELAMPUS_OPENAI_KEY, 'openai'))
	for _, engine in ipairs({ '', 'mlx', 'ollama', 'openai' }) do
		t.isFalse(visibleFor(fields.MELAMPUS_ANTHROPIC_KEY, engine), 'the Claude key field shows for ' .. engine)
	end
	t.isTrue(visibleFor(fields.MELAMPUS_ANTHROPIC_KEY, 'claude'))
end)

t.test('the password fields are not bound to the preferences', function()
	local contents = openSettings({ detection = detection() })
	for variable, entry in pairs(keyFields(contents)) do
		t.isNotNil(entry.view.bind_to_object, variable .. ' inherits the dialog\'s binding target, the preferences')
		t.isFalse(entry.view.bind_to_object == mock.state.prefs, variable .. ' is bound to the preferences')
	end
end)

--- Every file under the plugin folder, read whole.
local function pluginFiles()
	local files = {}
	local listing = io.popen('ls -1 "' .. PLUGIN .. '"')
	for name in listing:lines() do
		local handle = io.open(PLUGIN .. '/' .. name, 'rb')
		if handle then
			files[name] = handle:read('*a')
			handle:close()
		end
	end
	listing:close()
	return files
end

local TYPED = 'typed-into-the-dialog-not-a-real-key-7f3a'

t.test('a typed key is stored through LrPasswords and lands nowhere else', function()
	openSettings({
		detection = detection(),
		prefs = { engine = 'openai' },
		onDialog = function(options)
			local fields = keyFields(options.contents)
			fields.MELAMPUS_OPENAI_KEY.view.bind_to_object.MELAMPUS_OPENAI_KEY = TYPED
		end,
	})
	t.equals(mock.state.passwords.MELAMPUS_OPENAI_KEY, TYPED, 'the key was not stored through LrPasswords')
	t.isNil(mock.state.passwords.MELAMPUS_ANTHROPIC_KEY, 'a key was stored for the other engine')
	for key, value in pairs(mock.state.prefs) do
		t.isFalse(value == TYPED, 'the key is in the preferences under ' .. tostring(key))
		t.isFalse(type(value) == 'string' and string.find(value, TYPED, 1, true) ~= nil,
			'the key is in the preferences under ' .. tostring(key))
	end
	for name, text in pairs(pluginFiles()) do
		t.isNil(string.find(text, TYPED, 1, true), 'the key was written into the plugin folder: ' .. name)
	end
	for _, line in ipairs(mock.state.logLines) do
		t.isNil(string.find(line, TYPED, 1, true), 'the key was logged: ' .. line)
	end
end)

t.test('a stored key is shown back in its field, and an emptied one is forgotten', function()
	local contents = openSettings({
		detection = detection(),
		passwords = { MELAMPUS_ANTHROPIC_KEY = TYPED },
		onDialog = function(options)
			local fields = keyFields(options.contents)
			t.equals(fields.MELAMPUS_ANTHROPIC_KEY.view.bind_to_object.MELAMPUS_ANTHROPIC_KEY, TYPED,
				'the stored key is not in the field')
			fields.MELAMPUS_ANTHROPIC_KEY.view.bind_to_object.MELAMPUS_ANTHROPIC_KEY = ''
		end,
	})
	t.isNotNil(contents)
	t.equals(mock.state.passwords.MELAMPUS_ANTHROPIC_KEY, '', 'clearing the field did not clear the store')
end)

-- ── the model download row (card #408) ─────────────────────────────────────
local function buttonsTitled(row, prefix)
	local out = {}
	for _, entry in ipairs(viewsOfKind(row, 'push_button')) do
		if type(entry.view.title) == 'string' and entry.view.title:sub(1, #prefix) == prefix then out[#out + 1] = entry.view end
	end
	return out
end

--- Whether a view of the row shows in the model's current phase, through its
--- visible binding's transform.
local function shownNow(view, model)
	local binding = view.visible
	t.isNotNil(binding, 'no visible binding on ' .. tostring(view.title))
	t.equals(bindingKey(binding), 'phase')
	return binding.transform(model.phase)
end

local function theButton(row, model, prefix, expectShown)
	local found = buttonsTitled(row, prefix)
	t.equals(#found, 1, 'expected one button titled ' .. prefix)
	t.equals(shownNow(found[1], model), expectShown, prefix .. ' shown in phase ' .. tostring(model.phase))
	return found[1]
end

t.test('with the model absent the row shows a Download button with the model\'s name and size', function()
	local contents = openSettings({ detection = detection() })
	local row, model = modelRow(contents)
	t.isNotNil(row, 'no download row')
	t.equals(model.phase, 'absent')
	local button = theButton(row, model, 'Download ' .. REPO .. ' (18.3 GB)', true)
	t.equals(type(button.action), 'function', 'the button does nothing')
	theButton(row, model, 'Installed', false)
	theButton(row, model, 'Remove', false)
	theButton(row, model, 'Cancel', false)
end)

t.test('when the hub could not be reached the button says the size is unknown', function()
	local contents = openSettings({ detection = detection(), status = modelStatus({ bytes_total = 'null' }) })
	local row, model = modelRow(contents)
	theButton(row, model, 'Download ' .. REPO .. ' (size unknown)', true)
end)

t.test('with the model present the row reads Installed, greyed, and offers Remove', function()
	local contents = openSettings({ detection = detection(), status = modelStatus({
		installed = 'true', bytes_done = '18300000000', path = '"/hf/hub/models--x--y/snapshots/abc"' }) })
	local row, model = modelRow(contents)
	t.equals(model.phase, 'installed')
	local installed = theButton(row, model, 'Installed', true)
	t.isFalse(installed.enabled, 'Installed should be greyed')
	theButton(row, model, 'Remove', true)
	theButton(row, model, 'Download ' .. REPO, false)
end)

t.test('Remove runs --remove-model and the row flips to the Download button', function()
	local contents = openSettings({ detection = detection(), status = modelStatus({ installed = 'true' }) })
	local row, model = modelRow(contents)
	theButton(row, model, 'Remove', true).action()
	mock.settle()
	t.equals(commandsRun('--remove-model'), 1, 'Remove did not run the executable')
	t.equals(model.phase, 'absent')
	t.equals(#dialogsShown(false), 0, 'a message was shown for a removal that worked')
end)

t.test('a refused removal shows the message and the model stays Installed', function()
	local contents = openSettings({ detection = detection(), status = modelStatus({ installed = 'true' }), removeCode = 3 })
	local row, model = modelRow(contents)
	theButton(row, model, 'Remove', true).action()
	mock.settle()
	t.equals(model.phase, 'installed')
	t.equals(#dialogsShown(false), 1, 'no message for a refused removal')
end)

t.test('the row shows when the picked engine is mlx, or the unset preference resolves to it', function()
	local contents = openSettings({ detection = detection() })
	local row = modelRow(contents)
	t.equals(bindingKey(row.visible), 'engine', 'the row is not shown by the engine')
	t.isTrue(row.visible.transform('mlx'))
	t.isTrue(row.visible.transform(''), 'the default on this Mac is mlx')
	for _, engine in ipairs({ 'ollama', 'openai', 'claude' }) do
		t.isFalse(row.visible.transform(engine), 'the row shows for ' .. engine)
	end
end)

t.test('clicking Download runs the executable with stdout redirected, the progress follows the file, done flips to Installed', function()
	local contents = openSettings({ detection = detection(), download = {
		lines = { 'progress 0 18300000000', 'progress 3100000000 18300000000', 'progress 18300000000 18300000000',
			'done /hf/hub/models--x--y/snapshots/abc' },
		code = 0,
	} })
	local row, model = modelRow(contents)
	theButton(row, model, 'Download ' .. REPO, true).action()
	-- In the mock's temp directory, as the CLI log is.
	local progress = mock.state.tempDir .. '/melampus-download.progress'
	local log = mock.state.tempDir .. '/melampus-download.log'
	t.equals(mock.state.executed[#mock.state.executed],
		"'" .. EXECUTABLE .. "' --download-model >'" .. progress .. "' 2>'" .. log .. "'")
	t.equals(model.phase, 'downloading')
	theButton(row, model, 'Cancel', true)
	theButton(row, model, 'Download ' .. REPO, false)
	local text = viewsOfKind(row, 'static_text')
	local bar = nil
	for _, entry in ipairs(text) do
		if bindingKey(entry.view.title) == 'progress' then bar = entry.view end
	end
	t.isNotNil(bar, 'no progress text bound to the poller')
	t.isTrue(shownNow(bar, model), 'the progress is not shown while downloading')
	mock.tick()
	mock.tick()
	t.equals(model.progress, '3.1 GB of 18.3 GB')
	local scope = mock.state.progressScopes[#mock.state.progressScopes]
	t.isNotNil(scope, 'no progress scope for Lightroom\'s own bar')
	t.isTrue(math.abs(scope.portions[#scope.portions] - 3100000000 / 18300000000) < 1e-9)
	mock.settle()
	t.equals(model.phase, 'installed')
	t.isTrue(scope.isDone)
	t.equals(#dialogsShown(false), 0, 'a message was shown for a download that worked')
end)

t.test('Cancel writes the marker at the path the status named', function()
	os.remove(CANCEL_PATH)
	local contents = openSettings({ detection = detection(), download = {
		lines = { 'progress 0 18300000000', 'progress 3100000000 18300000000', 'cancelled' }, code = 4,
	} })
	local row, model = modelRow(contents)
	theButton(row, model, 'Download ' .. REPO, true).action()
	mock.tick()
	theButton(row, model, 'Cancel', true).action()
	local handle = io.open(CANCEL_PATH, 'r')
	t.isNotNil(handle, 'the marker was not written at ' .. CANCEL_PATH)
	if handle then handle:close() end
	mock.settle()
	t.equals(model.phase, 'absent', 'a cancelled download should offer Download again')
	t.equals(#dialogsShown(false), 0, 'a cancel is not an error')
	os.remove(CANCEL_PATH)
end)

t.test('a failed download shows a message with the tail of the log and offers Download again', function()
	local contents = openSettings({ detection = detection(), download = {
		lines = { 'progress 0 18300000000' }, code = 3,
		stderr = 'could not reach the hub at http://127.0.0.1:1: check the network\n',
	} })
	local row, model = modelRow(contents)
	theButton(row, model, 'Download ' .. REPO, true).action()
	mock.settle()
	t.equals(model.phase, 'absent')
	t.equals(#dialogsShown(false), 1, 'no message for a failed download')
	t.isNotNil(string.find(dialogsShown(false)[1].body, 'could not reach the hub', 1, true),
		'the message lacks the log tail: ' .. tostring(dialogsShown(false)[1].body))
	t.isNotNil(string.find(dialogsShown(false)[1].body, 'exit 3', 1, true))
end)

-- ── no executable ──────────────────────────────────────────────────────────
t.test('with no executable beside the plugin the dialog still opens, nothing greyed, and says why', function()
	local contents = openSettings({ detection = nil })
	t.isNil(mock.state.executed, 'ran a command with no executable to run')
	local picker = enginePicker(contents)
	t.isNotNil(picker, 'no picker')
	t.equals(#picker.items, 5)
	for _, item in ipairs(picker.items) do
		t.isTrue(item.enabled, item.value .. ' was greyed with no detection to grey it')
	end
	t.isTrue(#titlesMatching(contents, PLUGIN) > 0, 'the missing-executable message does not name the plugin folder')
	t.isTrue(#titlesMatching(contents, 'melampus') > 0, 'the missing-executable message does not name the file')
	t.isNil(modelRow(contents), 'a download row with no executable to download with')
end)

-- ── the download plumbing (card #408) ──────────────────────────────────────
-- LrTasks.execute blocks and returns only the exit code, so the download runs
-- in its own task with stdout redirected to a file, a second task reads that
-- file every second, and Cancel writes the marker the executable watches.
-- The mock steps the two tasks: each tick the fake executable writes one
-- more line and the poller reads what is there.

local function loadAnalyze(options)
	options = options or {}
	mock.reset({ existing = options.existing })
	mock.install(PLUGIN, options)
	for _, name in ipairs({ 'MelampusJson', 'MelampusRules', 'MelampusLog', 'MelampusAnalyze' }) do
		package.loaded[name] = nil
	end
	return dofile(PLUGIN .. '/MelampusAnalyze.lua')
end

local function append(path, text)
	local handle = assert(io.open(path, 'a'))
	handle:write(text)
	handle:close()
end

local function exists(path)
	local handle = io.open(path, 'r')
	if handle then handle:close() return true end
	return false
end

t.test('the download command runs the executable with stdout to the progress file and stderr to the log, on both shells', function()
	local Analyze = loadAnalyze({ existing = { [EXECUTABLE] = true } })
	local progress, log = Analyze.downloadFiles()
	t.equals(Analyze.downloadCommand(),
		"'" .. EXECUTABLE .. "' --download-model >'" .. progress .. "' 2>'" .. log .. "'")

	local exe = PLUGIN .. '\\melampus.exe'
	Analyze = loadAnalyze({ windows = true, existing = { [exe] = true } })
	progress, log = Analyze.downloadFiles()
	t.equals(Analyze.downloadCommand(),
		'""' .. exe .. '" --download-model >"' .. progress .. '" 2>"' .. log .. '""')
	t.isNotNil(string.find(progress, 'AppData\\Local\\Temp\\', 1, true), 'the progress file is not under temp: ' .. progress)
end)

t.test('without the executable the download command is the missing-executable message', function()
	local Analyze = loadAnalyze()
	local command, message = Analyze.downloadCommand()
	t.isNil(command)
	t.isNotNil(string.find(message, 'melampus', 1, true))
	t.isNotNil(string.find(message, PLUGIN, 1, true))
end)

--- Start a download under the mock whose fake executable writes `lines` to
--- the progress file one per tick, `stderr` to the log, and exits `code`.
local function startDownload(lines, code, stderr)
	local Analyze = loadAnalyze({ existing = { [EXECUTABLE] = true } })
	local progressFile, logFile = Analyze.downloadFiles()
	mock.state.onExecute = function(command)
		-- One line per tick, each after the poller has had its turn.
		for _, line in ipairs(lines) do
			mock.yield()
			append(progressFile, line .. '\n')
		end
		if stderr then append(logFile, stderr) end
		return code
	end
	local seen, finished = {}, nil
	local handle, err = Analyze.downloadModel(os.getenv('TMPDIR') .. '/melampus-data/cache/download-cancel',
		function(update) seen[#seen + 1] = update end,
		function(exit, update, tail) finished = { code = exit, update = update, tail = tail } end)
	t.isNotNil(handle, 'the download did not start: ' .. tostring(err))
	return seen, function() return finished end, handle
end

t.test('the poller reports each progress line as it lands in the file, then the exit with the last line', function()
	local seen, finished = startDownload({ 'progress 0 100', 'progress 40 100', 'progress 100 100',
		'done /hf/hub/models--x--y/snapshots/abc' }, 0)
	t.equals(#mock.state.executed, 1, 'the download command did not run')
	t.isNotNil(string.find(mock.state.executed[1], '--download-model', 1, true))
	mock.tick()
	t.equals(#seen, 1, 'after one tick, one line read')
	t.equals(seen[1].bytesDone, 0)
	mock.tick()
	t.equals(seen[#seen].bytesDone, 40)
	t.isNil(finished(), 'finished before the executable exited')
	mock.settle()
	t.equals(finished().code, 0)
	t.equals(finished().update.state, 'done')
	t.equals(finished().update.path, '/hf/hub/models--x--y/snapshots/abc')
	local counts = {}
	for _, update in ipairs(seen) do if update.state == 'progress' then counts[#counts + 1] = update.bytesDone end end
	t.equals(table.concat(counts, ','), '0,40,100')
	t.equals(#mock.state.tasks, 0, 'tasks left running after the download finished')
end)

t.test('a download that starts from a previous run\'s file does not read stale lines', function()
	local Analyze = loadAnalyze({ existing = { [EXECUTABLE] = true } })
	local progressFile = Analyze.downloadFiles()
	local handle = assert(io.open(progressFile, 'w'))
	handle:write('done /previous/run\n')
	handle:close()
	local seen, finished = startDownload({ 'progress 0 100' }, 4)
	mock.settle()
	t.equals(seen[1].state, 'progress', 'the previous run\'s done line was read')
	t.equals(finished().code, 4)
end)

t.test('Cancel writes the marker where the status said, creating its folder', function()
	local marker = os.getenv('TMPDIR') .. '/melampus-data/cache/download-cancel'
	os.remove(marker)
	local seen, finished, handle = startDownload({ 'progress 0 100', 'progress 10 100', 'cancelled' }, 4)
	mock.tick()
	handle.cancel()
	t.isTrue(exists(marker), 'the marker was not written at ' .. marker)
	mock.settle()
	t.equals(finished().code, 4)
	t.equals(finished().update.state, 'cancelled')
	os.remove(marker)
end)

t.test('a failed download hands back exit 3 and the tail of the log', function()
	local Analyze = loadAnalyze({ existing = { [EXECUTABLE] = true } })
	local _, logFile = Analyze.downloadFiles()
	os.remove(logFile)
	local seen, finished = startDownload({ 'progress 0 100' }, 3,
		'a warning first\ncould not reach the hub at http://127.0.0.1:1: check the network\n')
	mock.settle()
	t.equals(finished().code, 3)
	t.isNotNil(string.find(finished().tail, 'could not reach the hub', 1, true), 'no log tail: ' .. tostring(finished().tail))
end)

return t.summary()
