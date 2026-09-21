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
local OLLAMA_MODEL = 'qwen3-vl:8b-instruct'
local CANCEL_PATH = (os.getenv('TMPDIR') or '/tmp') .. '/melampus-data/cache/download-cancel'
local OLLAMA_UP = { available = true, reason = 'Ollama is answering at http://127.0.0.1:11434' }

--- The engines with a model to fetch (card #409), each with the name the
--- status reports, what the button says while it is absent (Ollama gives no
--- size for a model it does not hold), and its size once present.
local MODEL_ENGINES = {
	{ engine = 'mlx', name = REPO, absentTitle = 'Download ' .. REPO .. ' (18.3 GB)', size = '18300000000' },
	{ engine = 'ollama', name = OLLAMA_MODEL, absentTitle = 'Download ' .. OLLAMA_MODEL .. ' (size unknown)', size = '6100000000' },
}

--- The executable's --model-status answer for `engine` (mlx by default):
--- absent, with the size from the hub for mlx and none for ollama.
local function modelStatus(overrides, engine)
	local status = {
		repo = '"' .. REPO .. '"', installed = 'false', bytes_total = '18300000000',
		bytes_done = '0', path = 'null', cancel_path = '"' .. CANCEL_PATH .. '"',
	}
	if engine == 'ollama' then
		status.repo, status.bytes_total = '"' .. OLLAMA_MODEL .. '"', 'null'
	end
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

local function readFile(path)
	local handle = io.open(path, 'r')
	if not handle then return nil end
	local text = handle:read('a')
	handle:close()
	return text
end

--- The fake executable playing --download-model, as the onExecute the mock
--- calls with the command: `lines` land in the progress file (the command's
--- stdout redirect) one per tick, each after the poller has had its turn,
--- `stderr` in the log (its stderr redirect), and it exits `code`. `startup`
--- is how many ticks it spends starting (the one-file unpack, the imports,
--- the hub's listing) before the first line, the progress file empty.
local function fakeDownload(lines, code, stderr, startup)
	return function(command)
		local target = string.match(command, ">'([^']+)'")
		local log = string.match(command, "2>'([^']+)'")
		for _ = 1, startup or 0 do mock.yield() end
		for _, line in ipairs(lines) do
			mock.yield()
			writeFile(target, line .. '\n', 'a')
		end
		if stderr then writeFile(log, stderr, 'a') end
		return code
	end
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
--- model absent), for every engine, or `options.statusFor[engine]` for one;
--- `options.download` plays --download-model: its `lines` land
--- in the progress file one per tick, `stderr` in the log, and it exits
--- `code`, after `startup` ticks with the file empty; `options.detectionCode` and `options.statusCode` are the exit
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
				local engine = string.match(command, "%-%-backend '(%w+)'")
				local byEngine = options.statusFor or {}
				writeFile(target, byEngine[engine] or options.status or modelStatus(nil, engine))
				return options.statusCode or 0
			elseif string.find(command, '--download-model', 1, true) and target then
				local download = options.download or { lines = {}, code = 0 }
				return fakeDownload(download.lines, download.code, download.stderr, download.startup)(command)
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

--- The rows of the engine group that hold a model's buttons: the ones bound
--- to a property table with a `phase`, each with the table.
local function modelRows(contents)
	local rows = {}
	for _, kind in ipairs({ 'row', 'column' }) do
		for _, entry in ipairs(viewsOfKind(contents, kind)) do
			local bound = entry.view.bind_to_object
			if type(bound) == 'table' and bound.phase ~= nil then rows[#rows + 1] = { view = entry.view, model = bound } end
		end
	end
	return rows
end

--- The model row shown while `engine` (mlx by default) is picked, through
--- its visible binding, with its property table. nil when there is none.
local function modelRow(contents, engine)
	for _, row in ipairs(modelRows(contents)) do
		local binding = row.view.visible
		if type(binding) == 'table' and binding.transform(engine or 'mlx') then return row.view, row.model end
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
	t.equals(commandsRun("--model-status --backend 'mlx'"), 1, 'the status is asked for the engine')
	t.equals(#mock.state.executed, 2, 'nothing but detection and the status should run')
end)

t.test('with Ollama answering too, the status is asked once per engine with a model (card #409)', function()
	local contents = openSettings({ detection = mock.detectionText({ ollama = OLLAMA_UP }) })
	t.equals(commandsRun("--model-status --backend 'mlx'"), 1)
	t.equals(commandsRun("--model-status --backend 'ollama'"), 1)
	t.equals(#mock.state.executed, 3, 'nothing but detection and the two statuses should run')
	t.equals(#modelRows(contents), 2, 'one row per engine with a model')
end)

t.test('where neither mlx nor ollama can run the model is not asked about and there is no download row', function()
	local contents = openSettings({ detection = mock.detectionText({ mlx = { available = false, reason = 'needs Apple Silicon' } }) })
	t.equals(commandsRun('--model-status'), 0, 'the model was asked about where it cannot run')
	t.equals(#mock.state.executed, 1)
	t.equals(#modelRows(contents), 0, 'a download row with no engine to use it')
end)

t.test('on a machine with Ollama and no MLX, only Ollama\'s model is asked about (card #409)', function()
	local contents = openSettings({ detection = mock.detectionText({ mlx = { available = false, reason = 'needs Apple Silicon' }, ollama = OLLAMA_UP }) })
	t.equals(commandsRun("--model-status --backend 'mlx'"), 0, 'the MLX model was asked about where it cannot run')
	t.equals(commandsRun("--model-status --backend 'ollama'"), 1)
	t.equals(#modelRows(contents), 1)
	local row = modelRow(contents, 'ollama')
	t.isNotNil(row, 'no row for ollama')
	t.isTrue(row.visible.transform(''), 'the default on this machine is ollama')
	t.isFalse(row.visible.transform('mlx'))
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
--- through its visible binding's transform. A view bound to another table
--- (the key field to the keys table, the download row to the model) sits in
--- a column bound to the preferences, so its visible binding must name the
--- preferences itself, with the SDK's own spelling (`bind_to_object`;
--- anything else the SDK ignores).
local function visibleFor(entry, engine)
	local binding = entry.view.visible or (entry.parent and entry.parent.visible)
	local name = tostring(entry.view.title or entry.view.kind)
	t.isNotNil(binding, 'no visible binding on ' .. name)
	t.equals(bindingKey(binding), 'engine', 'the view is not shown by the engine: ' .. name)
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
--- visible binding's transform. The binding names the model table with the
--- SDK's own spelling (`bind_to_object`; anything else the SDK ignores).
local function shownNow(view, model)
	local binding = view.visible
	t.isNotNil(binding, 'no visible binding on ' .. tostring(view.title))
	t.equals(bindingKey(binding), 'phase')
	t.isTrue(binding.bind_to_object == model, 'the visible binding does not name the model as bind_to_object')
	return binding.transform(model.phase)
end

--- The progress scope the Download click most recently opened for
--- Lightroom's own bar, asserted present.
local function theScope()
	local scope = mock.state.progressScopes[#mock.state.progressScopes]
	t.isNotNil(scope, 'no progress scope for Lightroom\'s own bar')
	return scope
end

local function theButton(row, model, prefix, expectShown)
	local found = buttonsTitled(row, prefix)
	t.equals(#found, 1, 'expected one button titled ' .. prefix)
	t.equals(shownNow(found[1], model), expectShown, prefix .. ' shown in phase ' .. tostring(model.phase))
	return found[1]
end

--- The same row for each engine with a model (card #409): opened with
--- Ollama answering so both rows exist, each shown for its engine.
for _, case in ipairs(MODEL_ENGINES) do
	local engine, NAME = case.engine, case.name
	local function open(options)
		options = options or {}
		options.detection = options.detection or mock.detectionText({ ollama = OLLAMA_UP })
		if options.status then
			options.statusFor = { [engine] = options.status }
			options.status = nil
		end
		return openSettings(options)
	end

	--- Open the dialog with the engine's model installed (`status` merged
	--- into `options`), click the row's Remove button and settle;
	--- `beforeTheClick(options)`, when given, plays what changes between the
	--- dialog opening and the click (the status the executable answers after
	--- the removal, say, in `options.statusFor[engine]`). Returns the row and
	--- the model.
	local function removeClicked(options, beforeTheClick)
		options = options or {}
		options.status = modelStatus({ installed = 'true' }, engine)
		local contents = open(options)
		local row, model = modelRow(contents, engine)
		if beforeTheClick then beforeTheClick(options) end
		theButton(row, model, 'Remove', true).action()
		mock.settle()
		return row, model
	end

	t.test(engine .. ': with the model absent the row shows a Download button with the model\'s name and size', function()
		local contents = open()
		local row, model = modelRow(contents, engine)
		t.isNotNil(row, 'no download row')
		t.equals(model.phase, 'absent')
		local button = theButton(row, model, case.absentTitle, true)
		t.equals(type(button.action), 'function', 'the button does nothing')
		theButton(row, model, 'Installed', false)
		theButton(row, model, 'Remove', false)
		theButton(row, model, 'Cancel', false)
	end)

	t.test(engine .. ': when the size could not be had the button says the size is unknown', function()
		local contents = open({ status = modelStatus({ bytes_total = 'null' }, engine) })
		local row, model = modelRow(contents, engine)
		theButton(row, model, 'Download ' .. NAME .. ' (size unknown)', true)
	end)

	t.test(engine .. ': with the model present the row reads Installed, greyed, and offers Remove', function()
		local contents = open({ status = modelStatus({
			installed = 'true', bytes_total = case.size, bytes_done = case.size, path = '"/somewhere"' }, engine) })
		local row, model = modelRow(contents, engine)
		t.equals(model.phase, 'installed')
		local installed = theButton(row, model, 'Installed', true)
		t.isFalse(installed.enabled, 'Installed should be greyed')
		theButton(row, model, 'Remove', true)
		theButton(row, model, 'Download ' .. NAME, false)
	end)

	t.test(engine .. ': Remove runs --remove-model for the engine and the row flips to the Download button', function()
		local _, model = removeClicked()
		t.equals(commandsRun("--remove-model --backend '" .. engine .. "'"), 1, 'Remove did not run the executable for ' .. engine)
		t.equals(commandsRun('--remove-model'), 1)
		t.equals(model.phase, 'absent')
		t.equals(#dialogsShown(false), 0, 'a message was shown for a removal that worked')
	end)

	t.test(engine .. ': a refused removal shows the message and the model stays Installed', function()
		local _, model = removeClicked({ removeCode = 3 })
		t.equals(model.phase, 'installed')
		t.equals(#dialogsShown(false), 1, 'no message for a refused removal')
	end)

	t.test(engine .. ': a refused removal that set the model aside shows the message and the row flips to Download', function()
		-- Codex review 6, finding 1. A removal that moves the model aside but
		-- cannot delete it exits 3 with the model gone from the cache: the
		-- status is asked again after a refused removal, so the row reads what
		-- the engine holds, and the refusal's message is still shown.
		local _, model = removeClicked({ removeCode = 3 }, function(options)
			options.statusFor[engine] = modelStatus(nil, engine)
		end)
		t.equals(#dialogsShown(false), 1, 'no message for a refused removal')
		t.equals(commandsRun("--model-status --backend '" .. engine .. "'"), 2, 'the status was not asked again after the refused removal')
		t.equals(model.phase, 'absent')
	end)

	t.test(engine .. ': a refused removal keeps the row\'s phase when the status cannot be asked after it, and shows the refusal alone', function()
		-- Claude review 14, code finding 1. The status asked again after a
		-- refused removal can itself fail (exit 1: the cache unreadable by
		-- then); the row then keeps the phase it had, the message shown is the
		-- removal's, and the status's own failure is not shown over it.
		local _, model = removeClicked({ removeCode = 3 }, function(options)
			options.statusCode = 1
		end)
		t.equals(commandsRun("--model-status --backend '" .. engine .. "'"), 2, 'the status was not asked again after the refused removal')
		t.equals(model.phase, 'installed', 'the row lost its phase to a status it could not ask')
		local shown = dialogsShown(false)
		t.equals(#shown, 1, 'expected one message, the removal\'s')
		t.isNotNil(string.find(shown[1].body, 'could not remove the model (exit 3)', 1, true),
			'the message is not the removal\'s: ' .. tostring(shown[1].body))
	end)

	t.test(engine .. ': the row shows when the picked engine is ' .. engine .. ', and for the unset preference only when it resolves to it', function()
		local contents = open()
		local row = modelRow(contents, engine)
		t.isTrue(visibleFor({ view = row }, engine))
		t.equals(visibleFor({ view = row }, ''), engine == 'mlx', 'the default on this Mac is mlx')
		for _, other in ipairs({ 'mlx', 'ollama', 'openai', 'claude' }) do
			if other ~= engine then t.isFalse(visibleFor({ view = row }, other), 'the row shows for ' .. other) end
		end
	end)

	t.test(engine .. ': clicking Download runs the executable for the engine with stdout redirected, the progress follows the file, done flips to Installed', function()
		local contents = open({ download = {
			lines = { 'progress 0 18300000000', 'progress 3100000000 18300000000', 'progress 18300000000 18300000000',
				'done ' .. NAME },
			code = 0,
		} })
		local row, model = modelRow(contents, engine)
		theButton(row, model, 'Download ' .. NAME, true).action()
		-- In the mock's temp directory, as the CLI log is.
		local progress = mock.state.tempDir .. '/melampus-download.progress'
		local log = mock.state.tempDir .. '/melampus-download.log'
		t.equals(mock.state.executed[#mock.state.executed],
			"'" .. mock.EXECUTABLE .. "' --download-model --backend '" .. engine .. "' >'" .. progress .. "' 2>'" .. log .. "'")
		t.equals(model.phase, 'downloading')
		theButton(row, model, 'Cancel', true)
		theButton(row, model, 'Download ' .. NAME, false)
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
		local scope = theScope()
		t.isTrue(math.abs(scope.portions[#scope.portions] - 3100000000 / 18300000000) < 1e-9)
		mock.settle()
		t.equals(model.phase, 'installed')
		t.isTrue(scope.isDone)
		t.equals(#dialogsShown(false), 0, 'a message was shown for a download that worked')
	end)

	t.test(engine .. ': Cancel writes the marker at the path the status named', function()
		os.remove(CANCEL_PATH)
		local contents = open({ download = {
			lines = { 'progress 0 18300000000', 'progress 3100000000 18300000000', 'cancelled' }, code = 4,
		} })
		local row, model = modelRow(contents, engine)
		theButton(row, model, 'Download ' .. NAME, true).action()
		mock.tick()
		theButton(row, model, 'Cancel', true).action()
		t.isTrue(exists(CANCEL_PATH), 'the marker was not written at ' .. CANCEL_PATH)
		mock.settle()
		t.equals(model.phase, 'absent', 'a cancelled download should offer Download again')
		t.equals(#dialogsShown(false), 0, 'a cancel is not an error')
		os.remove(CANCEL_PATH)
	end)

	t.test(engine .. ': Lightroom\'s own cancel on the progress bar writes the marker too', function()
		-- The scope is cancelable, so the user can cancel from Lightroom's own
		-- progress bar as well as from the row; the poller asks the scope on
		-- each update and cancels the download the same way.
		os.remove(CANCEL_PATH)
		local contents = open({ download = {
			lines = { 'progress 0 18300000000', 'progress 3100000000 18300000000', 'cancelled' }, code = 4,
		} })
		local row, model = modelRow(contents, engine)
		theButton(row, model, 'Download ' .. NAME, true).action()
		t.isFalse(exists(CANCEL_PATH), 'a marker before anything was cancelled')
		mock.state.cancelled = true
		mock.tick()
		t.isTrue(exists(CANCEL_PATH), 'the progress bar\'s cancel did not write the marker at ' .. CANCEL_PATH)
		mock.settle()
		t.equals(model.phase, 'absent', 'a cancelled download should offer Download again')
		t.equals(#dialogsShown(false), 0, 'a cancel is not an error')
		os.remove(CANCEL_PATH)
	end)

	t.test(engine .. ': Lightroom\'s own cancel while the executable is still starting, the progress file empty, writes the marker on the next tick', function()
		-- Codex review 3, finding 2 (MelampusSettings.lua:127). The scope's
		-- cancel was asked only when a protocol line had arrived, so a cancel on
		-- Lightroom's bar during the executable's start-up (the unpack, the
		-- imports, the hub's listing: seconds with nothing in the progress file)
		-- wrote no marker until progress did. The poller asks on every tick.
		os.remove(CANCEL_PATH)
		local contents = open({ download = {
			startup = 3, lines = { 'cancelled' }, code = 4,
		} })
		local row, model = modelRow(contents, engine)
		theButton(row, model, 'Download ' .. NAME, true).action()
		local progress = mock.state.tempDir .. '/melampus-download.progress'
		mock.state.cancelled = true
		mock.tick()
		t.equals(readFile(progress), '', 'a protocol line reached the file before the tick')
		t.equals(model.progress, 'Starting…', 'progress reached the dialog before the tick')
		t.isTrue(exists(CANCEL_PATH), 'the progress bar\'s cancel with nothing in the progress file did not write the marker at ' .. CANCEL_PATH)
		mock.settle()
		t.equals(model.phase, 'absent', 'a cancelled download should offer Download again')
		t.equals(#dialogsShown(false), 0, 'a cancel is not an error')
		os.remove(CANCEL_PATH)
	end)

	t.test(engine .. ': a failed download shows a message with the tail of the log and offers Download again', function()
		local contents = open({ download = {
			lines = { 'progress 0 18300000000' }, code = 3,
			stderr = 'could not reach the hub at http://127.0.0.1:1: check the network\n',
		} })
		local row, model = modelRow(contents, engine)
		theButton(row, model, 'Download ' .. NAME, true).action()
		mock.settle()
		t.equals(model.phase, 'absent')
		t.equals(#dialogsShown(false), 1, 'no message for a failed download')
		t.isNotNil(string.find(dialogsShown(false)[1].body, 'could not reach the hub', 1, true),
			'the message lacks the log tail: ' .. tostring(dialogsShown(false)[1].body))
		t.isNotNil(string.find(dialogsShown(false)[1].body, 'exit 3', 1, true))
	end)
end

t.test('a Download clicked for the other engine while one download runs is refused, naming the engine still downloading, and the first goes on', function()
	-- Codex review (opposing vendor) round 1, code finding 1
	-- (MelampusSettings.lua:118). Both rows share one progress file, one log
	-- (Analyze.downloadFiles) and one cancel marker (--model-status's
	-- cancel_path), so a Download clicked on the other engine's row, the
	-- picker switched mid-download, started a second command that truncated
	-- the first's files; both pollers then read the second's progress, and
	-- Cancel on either row stopped whichever was running. One download at a
	-- time: the second click runs nothing, says which engine's model is still
	-- downloading, its row offers Download again, and the first download goes
	-- on to Installed. Proven in both orders, mlx then ollama and the reverse.
	for _, order in ipairs({ { MODEL_ENGINES[1], MODEL_ENGINES[2] }, { MODEL_ENGINES[2], MODEL_ENGINES[1] } }) do
		local first, second = order[1], order[2]
		local contents = openSettings({
			detection = mock.detectionText({ ollama = OLLAMA_UP }),
			download = { lines = { 'progress 0 100', 'progress 40 100', 'progress 100 100', 'done ' .. first.name }, code = 0 },
		})
		local firstRow, firstModel = modelRow(contents, first.engine)
		local secondRow, secondModel = modelRow(contents, second.engine)
		theButton(firstRow, firstModel, 'Download ' .. first.name, true).action()
		mock.tick()
		t.equals(firstModel.phase, 'downloading')
		-- The picker switched to the other engine: its row shows, with Download.
		theButton(secondRow, secondModel, 'Download ' .. second.name, true).action()
		t.equals(commandsRun('--download-model'), 1, 'a second download ran while the first was running (' .. first.engine .. ' then ' .. second.engine .. ')')
		t.equals(secondModel.phase, 'absent', 'the refused row should offer Download again')
		t.equals(firstModel.phase, 'downloading', 'the first download was disturbed')
		local shown = dialogsShown(false)
		t.equals(#shown, 1, 'expected one message, the refusal')
		t.isNotNil(string.find(shown[1].body, first.engine, 1, true),
			'the refusal does not name the engine still downloading: ' .. tostring(shown[1].body))
		t.isNotNil(string.find(shown[1].body, 'still downloading', 1, true), tostring(shown[1].body))
		mock.settle()
		t.equals(firstModel.phase, 'installed', 'the first download did not finish')
		t.equals(secondModel.phase, 'absent')
		t.equals(commandsRun('--download-model'), 1)
		t.equals(readFile(mock.state.tempDir .. '/melampus-download.progress'), 'progress 0 100\nprogress 40 100\nprogress 100 100\ndone ' .. first.name .. '\n',
			'the progress file is not the first download\'s alone')
	end
end)

t.test('a download that cannot start, the executable gone since the dialog opened, says so once and offers Download again', function()
	local contents = openSettings({ detection = mock.detectionText() })
	local row, model = modelRow(contents)
	-- The executable was beside the plugin when the dialog opened; by the
	-- click it is gone. Absent by name, so the disk is not consulted.
	mock.state.existing[mock.EXECUTABLE] = false
	theButton(row, model, 'Download ' .. REPO, true).action()
	t.equals(commandsRun('--download-model'), 0, 'ran a download with no executable to run it')
	t.equals(model.phase, 'absent', 'a download that could not start should offer Download again')
	local scope = theScope()
	t.isTrue(scope.isDone, 'the progress bar was left up with nothing to download')
	t.equals(#dialogsShown(false), 1, 'expected one message for a download that could not start')
	t.isNotNil(string.find(dialogsShown(false)[1].body, 'a file named melampus:', 1, true),
		'the message does not name the executable: ' .. tostring(dialogsShown(false)[1].body))
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
	t.equals(#modelRows(contents), 0, 'a download row with no executable to download with')
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

t.test('the download command runs the executable for the engine with stdout to the progress file and stderr to the log, on both shells', function()
	for _, engine in ipairs({ 'mlx', 'ollama' }) do
		local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
		local progress, log = Analyze.downloadFiles()
		t.equals(Analyze.downloadCommand(engine),
			"'" .. mock.EXECUTABLE .. "' --download-model --backend '" .. engine .. "' >'" .. progress .. "' 2>'" .. log .. "'")

		local exe = PLUGIN .. '\\melampus.exe'
		Analyze = loadAnalyze({ windows = true, existing = { [exe] = true } })
		progress, log = Analyze.downloadFiles()
		t.equals(Analyze.downloadCommand(engine),
			'""' .. exe .. '" --download-model --backend "' .. engine .. '" >"' .. progress .. '" 2>"' .. log .. '""')
		t.isNotNil(string.find(progress, 'AppData\\Local\\Temp\\', 1, true), 'the progress file is not under temp: ' .. progress)
	end
end)

t.test('the model commands refuse an engine the plugin does not know, before the shell', function()
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	local command, message = Analyze.downloadCommand('anthropic')
	t.isNil(command, 'an unknown engine reached the command line')
	t.isNotNil(string.find(message, 'anthropic', 1, true))
	local ok, why = Analyze.removeModel('anthropic')
	t.isFalse(ok)
	t.isNotNil(string.find(why, 'anthropic', 1, true))
	t.isNil(mock.state.executed, 'the executable ran for an unknown engine')
end)

t.test('without the executable the download command is the missing-executable message', function()
	-- Absent by name: the mock must not look at the disk, where the readme's
	-- install step may have put the real one.
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = false } })
	local command, message = Analyze.downloadCommand('mlx')
	t.isNil(command)
	t.isNotNil(string.find(message, 'melampus', 1, true))
	t.isNotNil(string.find(message, PLUGIN, 1, true))
end)

--- Start a download under the mock whose fake executable writes `lines` to
--- the progress file one per tick, `stderr` to the log, and exits `code`.
local function startDownload(lines, code, stderr)
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	mock.state.onExecute = fakeDownload(lines, code, stderr)
	local seen, finished = {}, nil
	local handle, err = Analyze.downloadModel('mlx', CANCEL_PATH,
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

t.test('a second download asked while one runs is refused, naming the engine, and starts once the first has exited', function()
	-- Codex review (opposing vendor) round 1, code finding 1
	-- (MelampusSettings.lua:118), at the controller: the progress file, the
	-- log and the marker are one set (downloadFiles, the status's
	-- cancel_path), so downloadModel runs one download at a time, whichever
	-- row asks; a second is nil plus a message naming the engine whose model
	-- is still downloading, nothing run, and the next one starts once the
	-- first has exited.
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	mock.state.onExecute = fakeDownload({ 'progress 0 100', 'done /hf/hub/models--x--y/snapshots/abc' }, 0)
	local finished = nil
	local handle = Analyze.downloadModel('mlx', CANCEL_PATH, function() end, function(exit) finished = exit end)
	t.isNotNil(handle, 'the first download did not start')
	mock.tick()
	local second, message = Analyze.downloadModel('ollama', CANCEL_PATH, function() end, function() end)
	t.isNil(second, 'a second download started while the first was running')
	t.isNotNil(string.find(tostring(message), 'mlx', 1, true), 'the refusal does not name the engine: ' .. tostring(message))
	t.equals(#mock.state.executed, 1, 'the second download ran the executable')
	mock.settle()
	t.equals(finished, 0)
	local third, why = Analyze.downloadModel('ollama', CANCEL_PATH, function() end, function() end)
	t.isNotNil(third, 'no download can start after the first exited: ' .. tostring(why))
	t.equals(#mock.state.executed, 2)
	mock.settle()
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
	local handle = Analyze.downloadModel('mlx', CANCEL_PATH, function() end,
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

t.test('a cancel asked elsewhere, Lightroom\'s own bar, is looked for on every tick, one with nothing in the progress file included', function()
	-- Codex review 3, finding 2. The dialog hands the poller a question,
	-- `cancelAsked`, for the scope's own Cancel; the poller asks it on every
	-- tick, before any protocol line exists, and once it says yes the cancel
	-- is held exactly as the handle's cancel() is.
	os.remove(CANCEL_PATH)
	local Analyze = loadAnalyze({ existing = { [mock.EXECUTABLE] = true } })
	mock.state.onExecute = fakeDownload({ 'cancelled' }, 4, nil, 2)
	local asked, seen, finished = false, {}, nil
	local handle = Analyze.downloadModel('mlx', CANCEL_PATH,
		function(update) seen[#seen + 1] = update end,
		function(exit, update) finished = { code = exit, update = update } end,
		function() return asked end)
	t.isNotNil(handle, 'the download did not start')
	asked = true
	mock.tick()
	t.equals(#seen, 0, 'a line reached the poller before the tick')
	t.isTrue(exists(CANCEL_PATH), 'a cancel asked elsewhere was not seen on a tick with nothing in the progress file')
	mock.settle()
	t.equals(finished.code, 4)
	t.equals(finished.update.state, 'cancelled')
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
