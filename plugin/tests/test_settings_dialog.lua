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

--- Open the real Settings dialog under the mock. `options.detection` is what
--- the executable prints for --detect-engines, from mock.detectionText (nil:
--- no executable beside the plugin, told to the mock as absent by name, so
--- the test holds after the readme's install step has put the real one
--- there); `options.onDialog` plays the user while the dialog is up.
local function openSettings(options)
	options = options or {}
	mock.loadUnderMock('MelampusSettings', {
		prefs = mock.defaultPrefs(options.prefs),
		existing = { [mock.EXECUTABLE] = options.detection ~= nil },
		passwords = options.passwords,
		onExecute = options.detection and mock.answersDetection(options.detection) or nil,
		onModalDialog = options.onDialog,
	})
	local modal = {}
	for _, dialog in ipairs(mock.state.dialogs) do
		if dialog.modal then modal[#modal + 1] = dialog end
	end
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

t.test('detection runs once, when the dialog opens', function()
	openSettings({ detection = mock.detectionText() })
	local detections = 0
	for _, command in ipairs(mock.state.executed or {}) do
		if string.find(command, '--detect-engines', 1, true) then detections = detections + 1 end
	end
	t.equals(detections, 1, 'the executable should be asked exactly once')
	t.equals(#mock.state.executed, 1, 'nothing but detection should run')
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
	-- satisfied by the folder line that :259 already checks.
	t.isTrue(#titlesMatching(contents, 'a file named melampus:') > 0, 'the missing-executable message does not name the file')
end)

return t.summary()
