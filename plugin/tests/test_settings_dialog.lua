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

local PLUGIN = mock.PLUGIN
local ENGINES = mock.loadPluginFile('MelampusRules').ENGINES
local OLLAMA_DOWNLOAD = 'https://ollama.com/download'

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

local function exists(path)
	local handle = io.open(path, 'r')
	if handle then handle:close() return true end
	return false
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
--- the executable prints for --detect-engines, from mock.detectionText (nil:
--- no executable beside the plugin, told to the mock as absent by name, so
--- the test holds after the readme's install step has put the real one
--- there); `options.status` what it prints for --model-status (default: the
--- model absent); `options.download` plays --download-model: its `lines` land
--- in the progress file one per tick, `stderr` in the log, and it exits
--- `code`; `options.detectionCode` and `options.statusCode` are the exit
--- codes of --detect-engines and --model-status (0); `options.removeCode` is
--- --remove-model's exit code; `options.onDialog` plays the user while the
--- dialog is up.
local function openSettings(options)
	options = options or {}
	mock.loadUnderMock('MelampusSettings', {
		prefs = mock.defaultPrefs(options.prefs),
		existing = { [mock.EXECUTABLE] = options.detection ~= nil },
		passwords = options.passwords,
		onExecute = function(command)
			local target = string.match(command, ">'([^']+)'")
			if string.find(command, '--detect-engines', 1, true) then
				if options.detection then return mock.answersDetection(options.detection, options.detectionCode)(command) end
			elseif string.find(command, '--model-status', 1, true) and target then
				writeFile(target, options.status or modelStatus())
				return options.statusCode or 0
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
	local modal = dialogsShown(true)
	t.equals(#modal, 1, 'expected exactly one dialog')
	return modal[1].contents
end

--- Every view of one kind in the tree, in order, each with its parent, from
--- the mock's walk of the recorded dialog.
local function viewsOfKind(root, kind)
	local found = {}
	for _, entry in ipairs(mock.views(root)) do
		if entry.view.kind == kind then found[#found + 1] = entry end
	end
	return found
end

local function bindingKey(binding)
	if type(binding) == 'table' then return binding.key end
	return binding
end

--- The one popup_menu bound to prefs.engine.
local function enginePicker(contents)
	return (mock.viewBoundTo(contents, 'engine'))
end

--- The static texts whose title contains `needle`.
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
	local contents = openSettings({ detection = mock.detectionText() })
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
	openSettings({ detection = mock.detectionText() })
	t.equals(commandsRun('--detect-engines'), 1, 'the executable should be asked once about the engines')
	t.equals(commandsRun('--model-status'), 1, 'the executable should be asked once about the model')
	t.equals(#mock.state.executed, 2, 'nothing but detection and the status should run')
end)

t.test('where mlx cannot run the model is not asked about and there is no download row', function()
	local contents = openSettings({ detection = mock.detectionText({ mlx = { available = false, reason = 'needs Apple Silicon' } }) })
	t.equals(commandsRun('--model-status'), 0, 'the model was asked about where it cannot run')
	t.equals(#mock.state.executed, 1)
	t.isNil(modelRow(contents), 'a download row with no mlx to use it')
end)

t.test('engines that cannot run here are greyed and their reasons shown', function()
	local contents = openSettings({ detection = mock.detectionText({ mlx = { available = false, reason = 'needs Apple Silicon' } }) })
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
	local contents = openSettings({ detection = mock.detectionText({ ollama = { available = true, reason = 'Ollama is answering at http://127.0.0.1:11434' } }) })
	for _, item in ipairs(enginePicker(contents).items) do
		t.isTrue(item.enabled, item.value .. ' was greyed')
	end
	t.equals(#titlesMatching(contents, 'Ollama'), 0, 'a reason was shown for an available engine')
end)

-- ── the Ollama link ────────────────────────────────────────────────────────
t.test('when ollama is unavailable a link opens the Ollama download page', function()
	local contents = openSettings({ detection = mock.detectionText() })
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
	local contents = openSettings({ detection = mock.detectionText({ ollama = { available = true, reason = 'Ollama is answering at http://127.0.0.1:11434' } }) })
	for _, entry in ipairs(viewsOfKind(contents, 'static_text')) do
		t.isNil(entry.view.mouse_down, 'a clickable link is shown with nothing to install: ' .. tostring(entry.view.title))
	end
	t.equals(#titlesMatching(contents, OLLAMA_DOWNLOAD), 0)
end)

t.test('another engine\'s reason naming an address gives no link while ollama is available', function()
	-- The link is Ollama's alone (Done-when 3): a reason from any other
	-- engine is shown under the picker as text, address and all, never as
	-- something to click.
	local MLX_ADDRESS = 'https://example.com/apple-silicon'
	local contents = openSettings({ detection = mock.detectionText({
		ollama = { available = true, reason = 'Ollama is answering at http://127.0.0.1:11434' },
		mlx = { available = false, reason = 'needs Apple Silicon; see ' .. MLX_ADDRESS },
	}) })
	for _, entry in ipairs(viewsOfKind(contents, 'static_text')) do
		t.isNil(entry.view.mouse_down, 'a clickable link is shown for an engine other than ollama: ' .. tostring(entry.view.title))
	end
	t.equals(#titlesMatching(contents, MLX_ADDRESS), 1, 'the mlx reason is shown once, as the note under the picker')
	t.equals(#mock.state.openedUrls, 0)
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
--- through its visible binding's transform. The row sits in a column bound
--- to the preferences, but the field beside it is bound to the keys table,
--- so the binding must name the preferences itself, with the SDK's own
--- spelling (`bind_to_object`; anything else the SDK ignores).
local function visibleFor(entry, engine)
	local binding = entry.view.visible or (entry.parent and entry.parent.visible)
	t.isNotNil(binding, 'the key field has no visible binding')
	t.equals(bindingKey(binding), 'engine', 'the key field is not shown by the engine')
	t.isTrue(binding.bind_to_object == mock.state.prefs, 'the visible binding does not name the preferences as bind_to_object')
	t.equals(type(binding.transform), 'function', 'the visible binding has no transform')
	return binding.transform(engine, mock.state.prefs)
end

t.test('a password field takes the key for openai and for claude, shown only when that engine is picked', function()
	local contents = openSettings({ detection = mock.detectionText() })
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
	local contents = openSettings({ detection = mock.detectionText() })
	for variable, entry in pairs(keyFields(contents)) do
		t.isNotNil(entry.view.bind_to_object, variable .. ' inherits the dialog\'s binding target, the preferences')
		t.isFalse(entry.view.bind_to_object == mock.state.prefs, variable .. ' is bound to the preferences')
	end
end)

--- Every file under the plugin folder, read whole.
local function pluginFiles()
	local files = {}
	local listing = io.popen('ls -1 ' .. mock.sh(PLUGIN))
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
		detection = mock.detectionText(),
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
		detection = mock.detectionText(),
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
	local contents = openSettings({ detection = mock.detectionText() })
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
	local contents = openSettings({ detection = mock.detectionText(), status = modelStatus({ bytes_total = 'null' }) })
	local row, model = modelRow(contents)
	theButton(row, model, 'Download ' .. REPO .. ' (size unknown)', true)
end)

t.test('with the model present the row reads Installed, greyed, and offers Remove', function()
	local contents = openSettings({ detection = mock.detectionText(), status = modelStatus({
		installed = 'true', bytes_done = '18300000000', path = '"/hf/hub/models--x--y/snapshots/abc"' }) })
	local row, model = modelRow(contents)
	t.equals(model.phase, 'installed')
	local installed = theButton(row, model, 'Installed', true)
	t.isFalse(installed.enabled, 'Installed should be greyed')
	theButton(row, model, 'Remove', true)
	theButton(row, model, 'Download ' .. REPO, false)
end)

t.test('Remove runs --remove-model and the row flips to the Download button', function()
	local contents = openSettings({ detection = mock.detectionText(), status = modelStatus({ installed = 'true' }) })
	local row, model = modelRow(contents)
	theButton(row, model, 'Remove', true).action()
	mock.settle()
	t.equals(commandsRun('--remove-model'), 1, 'Remove did not run the executable')
	t.equals(model.phase, 'absent')
	t.equals(#dialogsShown(false), 0, 'a message was shown for a removal that worked')
end)

t.test('a refused removal shows the message and the model stays Installed', function()
	local contents = openSettings({ detection = mock.detectionText(), status = modelStatus({ installed = 'true' }), removeCode = 3 })
	local row, model = modelRow(contents)
	theButton(row, model, 'Remove', true).action()
	mock.settle()
	t.equals(model.phase, 'installed')
	t.equals(#dialogsShown(false), 1, 'no message for a refused removal')
end)

t.test('the row shows when the picked engine is mlx, or the unset preference resolves to it', function()
	local contents = openSettings({ detection = mock.detectionText() })
	local row = modelRow(contents)
	t.equals(bindingKey(row.visible), 'engine', 'the row is not shown by the engine')
	t.isTrue(row.visible.transform('mlx'))
	t.isTrue(row.visible.transform(''), 'the default on this Mac is mlx')
	for _, engine in ipairs({ 'ollama', 'openai', 'claude' }) do
		t.isFalse(row.visible.transform(engine), 'the row shows for ' .. engine)
	end
end)

t.test('clicking Download runs the executable with stdout redirected, the progress follows the file, done flips to Installed', function()
	local contents = openSettings({ detection = mock.detectionText(), download = {
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
		"'" .. mock.EXECUTABLE .. "' --download-model >'" .. progress .. "' 2>'" .. log .. "'")
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
	local contents = openSettings({ detection = mock.detectionText(), download = {
		lines = { 'progress 0 18300000000', 'progress 3100000000 18300000000', 'cancelled' }, code = 4,
	} })
	local row, model = modelRow(contents)
	theButton(row, model, 'Download ' .. REPO, true).action()
	mock.tick()
	theButton(row, model, 'Cancel', true).action()
	t.isTrue(exists(CANCEL_PATH), 'the marker was not written at ' .. CANCEL_PATH)
	mock.settle()
	t.equals(model.phase, 'absent', 'a cancelled download should offer Download again')
	t.equals(#dialogsShown(false), 0, 'a cancel is not an error')
	os.remove(CANCEL_PATH)
end)

t.test('Lightroom\'s own cancel on the progress bar writes the marker too', function()
	-- The scope is cancelable, so the user can cancel from Lightroom's own
	-- progress bar as well as from the row; the poller asks the scope on
	-- each update and cancels the download the same way.
	os.remove(CANCEL_PATH)
	local contents = openSettings({ detection = mock.detectionText(), download = {
		lines = { 'progress 0 18300000000', 'progress 3100000000 18300000000', 'cancelled' }, code = 4,
	} })
	local row, model = modelRow(contents)
	theButton(row, model, 'Download ' .. REPO, true).action()
	t.isFalse(exists(CANCEL_PATH), 'a marker before anything was cancelled')
	mock.state.cancelled = true
	mock.tick()
	t.isTrue(exists(CANCEL_PATH), 'the progress bar\'s cancel did not write the marker at ' .. CANCEL_PATH)
	mock.settle()
	t.equals(model.phase, 'absent', 'a cancelled download should offer Download again')
	t.equals(#dialogsShown(false), 0, 'a cancel is not an error')
	os.remove(CANCEL_PATH)
end)

t.test('a failed download shows a message with the tail of the log and offers Download again', function()
	local contents = openSettings({ detection = mock.detectionText(), download = {
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

-- ── a command that fails ───────────────────────────────────────────────────
t.test('when the status exits non-zero the dialog says it could not ask about the model, and there is no row', function()
	local contents = openSettings({ detection = mock.detectionText(), statusCode = 1 })
	t.equals(#titlesMatching(contents, 'Melampus could not ask its analysis program about the model (exit 1).'), 1,
		'the message does not read as a sentence: ' .. table.concat(mock.dialogStrings(contents), ' | '))
	t.isNil(modelRow(contents), 'a download row with no status to build it from')
end)

t.test('when detection exits non-zero the note under the picker says it could not ask about the engines', function()
	local contents = openSettings({ detection = mock.detectionText(), detectionCode = 1 })
	t.equals(#titlesMatching(contents, 'Melampus could not ask its analysis program about the engines (exit 1).'), 1,
		'the note does not read as a sentence: ' .. table.concat(mock.dialogStrings(contents), ' | '))
	for _, item in ipairs(enginePicker(contents).items) do
		t.isTrue(item.enabled, item.value .. ' was greyed with no detection to grey it')
	end
end)

t.test('a status the dialog does not understand is said to be about the model, and names the file', function()
	local contents = openSettings({ detection = mock.detectionText(), status = 'not json' })
	local notes = titlesMatching(contents, 'Melampus did not understand what its analysis program said about the model')
	t.equals(#notes, 1, table.concat(mock.dialogStrings(contents), ' | '))
	t.isNotNil(string.find(notes[1].title, 'melampus-model-status.json', 1, true), 'the message does not name the file')
	t.isNil(modelRow(contents))
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
	-- The file as the message says it, not the bare word: the folder's own path
	-- holds "melampus" wherever the repository lives, so the word alone is
	-- satisfied by the folder assertion just above.
	t.isTrue(#titlesMatching(contents, 'a file named melampus:') > 0, 'the missing-executable message does not name the file')
	t.isNil(modelRow(contents), 'a download row with no executable to download with')
end)

-- ── the download plumbing (card #408) ──────────────────────────────────────
-- LrTasks.execute blocks and returns only the exit code, so the download runs
-- in its own task with stdout redirected to a file, a second task reads that
-- file every second, and Cancel writes the marker the executable watches.
-- The mock steps the two tasks: each tick the fake executable writes one
-- more line and the poller reads what is there.

--- MelampusAnalyze.lua fresh under the mock, through its loader: `existing`
--- tells the mock which files are there; the rest are install options.
local function loadAnalyze(options)
	options = options or {}
	return mock.loadUnderMock('MelampusAnalyze', { existing = options.existing }, PLUGIN, options)
end

t.test('the download command runs the executable with stdout to the progress file and stderr to the log, on both shells', function()
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	local progress, log = Analyze.downloadFiles()
	t.equals(Analyze.downloadCommand(),
		"'" .. mock.EXECUTABLE .. "' --download-model >'" .. progress .. "' 2>'" .. log .. "'")

	local exe = PLUGIN .. '\\melampus.exe'
	Analyze = loadAnalyze({ windows = true, existing = { [exe] = true } })
	progress, log = Analyze.downloadFiles()
	t.equals(Analyze.downloadCommand(),
		'""' .. exe .. '" --download-model >"' .. progress .. '" 2>"' .. log .. '""')
	t.isNotNil(string.find(progress, 'AppData\\Local\\Temp\\', 1, true), 'the progress file is not under temp: ' .. progress)
end)

t.test('without the executable the download command is the missing-executable message', function()
	-- Absent by name: the mock must not look at the disk, where the readme's
	-- install step may have put the real one.
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = false } })
	local command, message = Analyze.downloadCommand()
	t.isNil(command)
	t.isNotNil(string.find(message, 'melampus', 1, true))
	t.isNotNil(string.find(message, PLUGIN, 1, true))
end)

--- Start a download under the mock whose fake executable writes `lines` to
--- the progress file one per tick, `stderr` to the log, and exits `code`.
local function startDownload(lines, code, stderr)
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	local progressFile, logFile = Analyze.downloadFiles()
	mock.state.onExecute = function(command)
		-- One line per tick, each after the poller has had its turn.
		for _, line in ipairs(lines) do
			mock.yield()
			writeFile(progressFile, line .. '\n', 'a')
		end
		if stderr then writeFile(logFile, stderr, 'a') end
		return code
	end
	local seen, finished = {}, nil
	local handle, err = Analyze.downloadModel(CANCEL_PATH,
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
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	local progressFile = Analyze.downloadFiles()
	writeFile(progressFile, 'done /previous/run\n')
	local seen, finished = startDownload({ 'progress 0 100' }, 4)
	mock.settle()
	t.equals(seen[1].state, 'progress', 'the previous run\'s done line was read')
	t.equals(finished().code, 4)
end)

t.test('Cancel writes the marker where the status said, creating its folder', function()
	os.remove(CANCEL_PATH)
	local seen, finished, handle = startDownload({ 'progress 0 100', 'progress 10 100', 'cancelled' }, 4)
	mock.tick()
	handle.cancel()
	t.isTrue(exists(CANCEL_PATH), 'the marker was not written at ' .. CANCEL_PATH)
	mock.settle()
	t.equals(finished().code, 4)
	t.equals(finished().update.state, 'cancelled')
	os.remove(CANCEL_PATH)
end)

t.test('a Cancel clicked while the executable is still starting holds: the marker comes back on the next tick', function()
	-- Done-when 2. The executable removes a stale marker when it starts, and
	-- its start (the one-file unpack, the imports) takes seconds after the
	-- click. A Cancel in that window must not be lost: the poller writes the
	-- marker again on every tick until the command exits.
	os.remove(CANCEL_PATH)
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	local progressFile = Analyze.downloadFiles()
	local markerAtStart = nil
	mock.state.onExecute = function()
		mock.yield()
		-- What download_model does first: the marker it finds is stale.
		markerAtStart = exists(CANCEL_PATH)
		os.remove(CANCEL_PATH)
		mock.yield()
		writeFile(progressFile, 'progress 0 100\n', 'a')
		mock.yield()
		writeFile(progressFile, 'cancelled\n', 'a')
		-- And on exit, whatever the outcome.
		os.remove(CANCEL_PATH)
		return 4
	end
	local finished = nil
	local handle = Analyze.downloadModel(CANCEL_PATH, function() end,
		function(exit, update) finished = { code = exit, update = update } end)
	handle.cancel()
	t.isTrue(exists(CANCEL_PATH), 'Cancel did not write the marker')
	mock.tick()
	t.isTrue(markerAtStart, 'the fake executable did not find the marker to remove')
	t.isTrue(exists(CANCEL_PATH), 'the marker the executable removed on start was not written again')
	mock.settle()
	t.equals(finished.code, 4)
	t.isFalse(exists(CANCEL_PATH), 'the poller kept writing the marker after the command exited')
	os.remove(CANCEL_PATH)
end)

t.test('a failed download hands back exit 3 and the tail of the log', function()
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	local _, logFile = Analyze.downloadFiles()
	os.remove(logFile)
	local seen, finished = startDownload({ 'progress 0 100' }, 3,
		'a warning first\ncould not reach the hub at http://127.0.0.1:1: check the network\n')
	mock.settle()
	t.equals(finished().code, 3)
	t.isNotNil(string.find(finished().tail, 'could not reach the hub', 1, true), 'no log tail: ' .. tostring(finished().tail))
end)

return t.summary()
