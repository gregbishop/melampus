--[[
A mock of the Lightroom Classic SDK, enough to execute the real plugin code.

This exists because the alternative was shipping Lua that had never run and
asking a human to be the test runner. It cannot catch genuine SDK behavioural
differences, but it does catch what actually went wrong repeatedly: nil
dereferences, wrong call signatures, misordered arguments, logic that silently
produces nothing, and counters that report success without writing.

The mock records every catalog mutation so tests can assert on what the plugin
*did*, not merely that it ran without error.
--]]

local M = {}

M.state = {}

--- Single-quote a path for sh. Every path the mock hands to a shell goes
-- through here: the temp directory comes from TMPDIR, and a space, a quote,
-- a "$" or a backtick in it must arrive as the name it is, not be split,
-- expanded or run.
--
-- The plugin's own quote() in MelampusAnalyze.lua has the same sh branch and
-- is not used here on purpose. The mock's shell is always the host's sh, even
-- while it fakes Windows; the plugin's quote() follows WIN_ENV, which the
-- last install left set when the next reset's cleanUp runs, so it would
-- double-quote for sh and "$" would expand again. And the mock stands in for
-- the SDK beneath the plugin: it is loaded before `import` exists, so no
-- plugin module can load yet, and its housekeeping must not depend on the
-- module it exists to exercise.
local function sh(text)
	return "'" .. string.gsub(tostring(text), "'", "'\\''") .. "'"
end
-- Exposed so a suite can spell what a sh line must hold, from the same
-- quoting the mock's own shell traffic uses, and still not from the plugin's.
M.sh = sh

--- Plays the executable for real, for state.onExecute: the line goes to the
-- host's shell the way LrTasks.execute hands it to Lightroom's (sh -c, or
-- cmd.exe /c on a Windows host), and the exit code comes back as a number,
-- which is what Lightroom's Lua 5.1 os.execute returns; 5.2 and later
-- return a boolean, the word 'exit' and then the code.
function M.runThroughTheShell(command)
	local first, _, code = os.execute(command)
	if type(first) == 'number' then return first end
	return code
end

--- Where Claude Code is installed from, as the not-installed reason says.
M.CLAUDE_CODE_INSTALL = 'https://code.claude.com/docs/en/setup'

--- What `melampus --detect-engines` says on a Mac with no Ollama running,
-- no Claude Code installed and a Codex CLI that is not signed in, decoded:
-- one verdict per engine, in the order the executable prints them, each
-- with the title the picker shows (card #423). `overrides[engine]` replaces
-- fields of that engine's verdict. The one canned answer every suite starts
-- from, so a title and a reason are spelled once; `M.canned[engine]` is the
-- same answer by engine, `M.titles[engine]` its title, and `M.signedIn` the
-- other answer the subscription CLIs give, as an overrides table.
function M.detectionVerdicts(overrides)
	local list = {
		{ engine = 'mlx', title = 'MLX — local, Apple Silicon', available = true,
			reason = 'runs locally on this Apple Silicon Mac' },
		{ engine = 'ollama', title = 'Ollama — local', available = false,
			reason = 'no Ollama server at http://127.0.0.1:11434; install it from https://ollama.com/download' },
		{ engine = 'openai', title = 'OpenAI — cloud, needs an API key', available = true,
			reason = 'API key required: set MELAMPUS_OPENAI_KEY (or OPENAI_API_KEY)' },
		{ engine = 'claude', title = 'Claude — cloud, needs an API key', available = true,
			reason = 'API key required: set MELAMPUS_ANTHROPIC_KEY (or ANTHROPIC_API_KEY)' },
		{ engine = 'claude-code', title = 'Claude Code — subscription, no API key', available = false,
			reason = "Claude Code is not installed: nothing on PATH is called 'claude'; "
				.. 'install it from ' .. M.CLAUDE_CODE_INSTALL .. ', then sign in with `claude auth login`' },
		{ engine = 'codex', title = 'Codex CLI — subscription, no API key', available = false,
			reason = 'Codex CLI is installed but not signed in; run `codex login`' },
	}
	for _, v in ipairs(list) do
		local o = overrides and overrides[v.engine]
		if o then for k, value in pairs(o) do v[k] = value end end
	end
	return list
end

M.canned = {}
M.titles = {}
for _, v in ipairs(M.detectionVerdicts()) do
	M.canned[v.engine] = v
	M.titles[v.engine] = v.title
end

--- The picker's items (Rules.engineItems, or a popup_menu's) indexed by
-- value, so a test can name one: M.itemsByValue(items).codex.
function M.itemsByValue(items)
	local byValue = {}
	for _, item in ipairs(items) do byValue[item.value] = item end
	return byValue
end

--- The subscription CLIs signed in (card #423): each available, with the
-- billing sentence providers._cli_verdict prints, the account in brackets.
M.signedIn = {
	['claude-code'] = { available = true,
		reason = 'Claude Code is signed in (claude.ai, max); every frame bills to that subscription, not to an API key' },
	codex = { available = true,
		reason = 'Codex CLI is signed in (ChatGPT); every frame bills to that subscription, not to an API key' },
}

--- The same answer as the JSON text the executable prints.
function M.detectionText(overrides)
	local function quoted(text)
		return '"' .. string.gsub(tostring(text), '[\\"]', '\\%0') .. '"'
	end
	local parts = {}
	for _, v in ipairs(M.detectionVerdicts(overrides)) do
		parts[#parts + 1] = string.format('{"engine": %s, "title": %s, "available": %s, "reason": %s}',
			quoted(v.engine), quoted(v.title), tostring(v.available), quoted(v.reason))
	end
	return '[' .. table.concat(parts, ', ') .. ']'
end

--- A fake executable for state.onExecute: answers a --detect-engines command
-- by writing `text` to the file the command's stdout is redirected to, and
-- exits with `code`; any other command exits 0 and writes nothing.
function M.answersDetection(text, code)
	return function(command)
		if not string.find(command, '--detect-engines', 1, true) then return 0 end
		local target = string.match(command, ">'([^']+)'")
		local handle = assert(io.open(target, 'w'))
		handle:write(text)
		handle:close()
		return code or 0
	end
end

--- Remove the temp directory this run made, if it made one.
function M.cleanUp()
	if M.state.tempDir then
		os.execute('rm -rf ' .. sh(M.state.tempDir))
		M.state.tempDir = nil
	end
end

function M.reset(options)
	options = options or {}
	M.cleanUp()
	M.state = {
		photos = {},
		keywords = {},        -- id -> { name, parent, children }
		nextKeywordId = 1,
		writeTransactions = {},
		privateTransactions = {},
		dialogs = {},
		confirmAnswer = options.confirmAnswer or 'ok',
		logLines = {},
		prefs = options.prefs or {},
		cancelled = false,
		yieldInsideWrite = false,
		inWriteGate = false,
		-- fileName -> how many write operations to accept-and-discard. Simulates the
		-- failure this project actually hit: a write that raises nothing and lands
		-- nothing, so a counter next to the call reports success that never happened.
		dropWrites = options.dropWrites or {},
		-- Paths LrFileUtils.exists answers for without touching the disk:
		-- true is present, false is absent. The executable beside the
		-- plugin, on either platform; false so a test that wants it missing
		-- holds after the readme's install step has put the real one there.
		existing = options.existing or {},
		-- Plays the executable for LrTasks.execute: given the command, it
		-- writes what the real one would and returns its exit code.
		onExecute = options.onExecute,
		-- Plays the user while a modal dialog is up: given the options, it
		-- can set values the way typing would.
		onModalDialog = options.onModalDialog,
		-- What LrPasswords holds, by key string; the URLs the browser was
		-- asked to open.
		passwords = options.passwords or {},
		openedUrls = {},
		-- The Windows temp folder a fake Windows Lightroom reports, when a
		-- test names one; otherwise windowsTemp() decides.
		windowsTemp = options.windowsTemp,
		-- How many previews the plugin asked for in this run.
		previewsRequested = 0,
		-- The async tasks started and not yet finished, as coroutines: a
		-- task runs until it sleeps, and M.tick() resumes every sleeping one
		-- once. The progress scopes created, with the portions they were set to.
		tasks = {},
		progressScopes = {},
	}
end

-- ── async tasks ────────────────────────────────────────────────────────────
-- Lightroom runs LrTasks.startAsyncTask functions cooperatively: they run
-- until they sleep or yield, and others run in between. The mock does the
-- same with coroutines, so a test can step a download and the poller that
-- reads its file one tick at a time. A task that never sleeps runs to
-- completion inside startAsyncTask, exactly as before.

local function inTask()
	local co, isMain = coroutine.running()
	return co ~= nil and not isMain
end

local function resume(task)
	local ok, err = coroutine.resume(task)
	if not ok then error(err, 0) end
end

--- Give the other tasks a turn, when called from inside one. What a fake
-- executable calls between the lines it writes, so the poller reads them
-- one at a time.
function M.yield()
	if inTask() then coroutine.yield() end
end

--- Resume every sleeping task once; returns how many are still alive.
function M.tick()
	local alive = {}
	for _, task in ipairs(M.state.tasks) do
		if coroutine.status(task) == 'suspended' then resume(task) end
		if coroutine.status(task) ~= 'dead' then alive[#alive + 1] = task end
	end
	M.state.tasks = alive
	return #alive
end

--- Tick until every task has finished, or `limit` ticks (default 1000).
function M.settle(limit)
	for _ = 1, limit or 1000 do
		if M.tick() == 0 then return end
	end
	error('tasks still running after ' .. tostring(limit or 1000) .. ' ticks')
end

-- ── keyword objects ────────────────────────────────────────────────────────
local Keyword = {}
Keyword.__index = Keyword

function Keyword:getName() return self.name end
function Keyword:getParent() return self.parent end
function Keyword:getChildren() return self.children end
function Keyword:getPhotos()
	local out = {}
	for _, photo in ipairs(M.state.photos) do
		for _, kw in ipairs(photo._keywords) do
			if kw == self then out[#out + 1] = photo end
		end
	end
	return out
end

-- ── photo objects ──────────────────────────────────────────────────────────
local Photo = {}
Photo.__index = Photo

function Photo:getRawMetadata(key)
	if key == 'keywords' then return self._keywords end
	return self._raw[key]
end

function Photo:getFormattedMetadata(key)
	if key == 'fileName' then return self.fileName end
	return self._formatted and self._formatted[key]
end

--- True when this write should be silently discarded, per state.dropWrites.
function Photo:_swallow()
	local remaining = M.state.dropWrites[self.fileName]
	if remaining == nil or remaining <= 0 then return false end
	M.state.dropWrites[self.fileName] = remaining - 1
	return true
end

function Photo:setRawMetadata(key, value)
	assert(M.state.inWriteGate, 'setRawMetadata outside a write gate')
	if self:_swallow() then return end
	self._raw[key] = value
	self._writes[#self._writes + 1] = { key = key, value = value }
end

function Photo:addKeyword(keyword)
	assert(M.state.inWriteGate, 'addKeyword outside a write gate')
	assert(keyword ~= nil, 'addKeyword(nil)')
	if self:_swallow() then return end
	for _, existing in ipairs(self._keywords) do
		if existing == keyword then return end
	end
	self._keywords[#self._keywords + 1] = keyword
end

function Photo:getPropertyForPlugin(_, field) return self._plugin[field] end

--- The real call is asynchronous and the plugin must retain the returned object;
-- the mock answers at once with bytes that are not a JPEG, which is enough for
-- the export loop to write a file and count it.
function Photo:requestJpegThumbnail(width, height, callback)
	M.state.previewsRequested = M.state.previewsRequested + 1
	callback('mock-preview-bytes', nil)
	return {}
end

function Photo:setPropertyForPlugin(_, field, value)
	assert(M.state.inWriteGate, 'setPropertyForPlugin outside a write gate')
	self._plugin[field] = value
end

--- Full "A > B > C" path of each keyword, for assertions.
function Photo:keywordPaths()
	local out = {}
	for _, kw in ipairs(self._keywords) do
		local parts, node = {}, kw
		while node do
			table.insert(parts, 1, node.name)
			node = node.parent
		end
		out[#out + 1] = table.concat(parts, ' > ')
	end
	table.sort(out)
	return out
end

function M.addPhoto(fileName, raw)
	local photo = setmetatable({
		fileName = fileName,
		_raw = raw or {},
		_keywords = {},
		_plugin = {},
		_writes = {},
	}, Photo)
	M.state.photos[#M.state.photos + 1] = photo
	return photo
end

-- ── catalog ────────────────────────────────────────────────────────────────
local Catalog = {}
Catalog.__index = Catalog

function Catalog:getTargetPhotos() return M.state.photos end
function Catalog:getAllPhotos() return M.state.photos end

function Catalog:getKeywords()
	local roots = {}
	for _, kw in ipairs(M.state.keywords) do
		if kw.parent == nil then roots[#roots + 1] = kw end
	end
	return roots
end

function Catalog:createKeyword(name, synonyms, includeOnExport, parent, returnExisting)
	assert(M.state.inWriteGate, 'createKeyword outside a write gate')
	local siblings = parent and parent.children or self:getKeywords()
	for _, kw in ipairs(siblings) do
		if kw.name == name then
			if returnExisting then return kw end
			return nil
		end
	end
	-- Reproduces the real trap: with returnExisting, a name that already exists
	-- elsewhere in the tree can come back nil rather than creating a new node.
	if returnExisting then
		for _, kw in ipairs(M.state.keywords) do
			if kw.name == name and kw.parent ~= parent then
				return nil
			end
		end
	end
	local kw = setmetatable({
		name = name, parent = parent, children = {},
		id = M.state.nextKeywordId,
	}, Keyword)
	M.state.nextKeywordId = M.state.nextKeywordId + 1
	M.state.keywords[#M.state.keywords + 1] = kw
	if parent then parent.children[#parent.children + 1] = kw end
	return kw
end

function Catalog:deleteKeyword(keyword)
	assert(M.state.inWriteGate, 'deleteKeyword outside a write gate')
	for index, kw in ipairs(M.state.keywords) do
		if kw == keyword then table.remove(M.state.keywords, index); break end
	end
end

function Catalog:createCollectionSet(name, parent, returnExisting) return { name = name } end
function Catalog:createSmartCollection(name, spec, parent, returnExisting)
	return { name = name, spec = spec }
end

function Catalog:withWriteAccessDo(name, func, options)
	M.state.writeTransactions[#M.state.writeTransactions + 1] = name
	M.state.inWriteGate = true
	local ok, err = pcall(func)
	M.state.inWriteGate = false
	if not ok then error(err, 0) end
end

function Catalog:withPrivateWriteAccessDo(func, options)
	M.state.privateTransactions[#M.state.privateTransactions + 1] = true
	M.state.inWriteGate = true
	local ok, err = pcall(func)
	M.state.inWriteGate = false
	if not ok then error(err, 0) end
end

local catalog = setmetatable({}, Catalog)

-- ── the namespaces the plugin imports ──────────────────────────────────────
local namespaces = {}

namespaces.LrApplication = { activeCatalog = function() return catalog end }

namespaces.LrDialogs = {
	message = function(title, body, kind)
		M.state.dialogs[#M.state.dialogs + 1] = { title = title, body = body, kind = kind }
	end,
	confirm = function(title, body, action, cancel)
		M.state.dialogs[#M.state.dialogs + 1] = { title = title, body = body, confirm = true }
		return M.state.confirmAnswer
	end,
	runOpenPanel = function() return nil end,
	runSavePanel = function() return nil end,
	-- Recorded with its view tree, so a test can read what the dialog says.
	presentModalDialog = function(options)
		M.state.dialogs[#M.state.dialogs + 1] = {
			title = options.title, contents = options.contents, modal = true }
		if M.state.onModalDialog then M.state.onModalDialog(options) end
		return 'ok'
	end,
}

-- ── reading a recorded dialog ──────────────────────────────────────────────
-- One walk of the view tree a dialog was presented with, for every suite that
-- reads one: children are the array part of each view, attributes the rest
-- (see LrView below).

--- Every view under `root`, the root first, in the order the plugin built
--- them, each as { view, parent, group }: `parent` the view holding it,
--- `group` the nearest group_box around it (itself, when it is one).
function M.views(root)
	local found = {}
	local function walk(node, parent, group)
		if type(node) ~= 'table' then return end
		if node.kind == 'group_box' then group = node end
		found[#found + 1] = { view = node, parent = parent, group = group }
		for _, child in ipairs(node) do walk(child, node, group) end
	end
	walk(root, nil, nil)
	return found
end

--- The first view whose value is bound to `key` (a `bind 'key'` is the key
--- itself under this mock), and the group box that holds it.
function M.viewBoundTo(root, key)
	for _, entry in ipairs(M.views(root)) do
		if entry.view.value == key then return entry.view, entry.group end
	end
	return nil
end

--- Every string a dialog shows: titles (static text, group boxes, checkboxes,
--- buttons) and tooltips, in order.
function M.dialogStrings(root)
	local out = {}
	for _, entry in ipairs(M.views(root)) do
		for _, key in ipairs({ 'title', 'tooltip' }) do
			if type(entry.view[key]) == 'string' then out[#out + 1] = entry.view[key] end
		end
	end
	return out
end

--- Stored and retrieved by key string; the real one keeps them in the OS
-- keychain on macOS, which is exactly why a test must see them land here and
-- nowhere else.
namespaces.LrPasswords = {
	store = function(key, password)
		assert(key ~= nil and password ~= nil, 'keystring or password is nil.')
		M.state.passwords[key] = password
	end,
	retrieve = function(key)
		assert(key ~= nil, 'keystring is nil.')
		return M.state.passwords[key]
	end,
}

namespaces.LrHttp = {
	openUrlInBrowser = function(url)
		M.state.openedUrls[#M.state.openedUrls + 1] = url
	end,
}

--- An observable property table is, for these tests, a plain table.
namespaces.LrBinding = {
	makePropertyTable = function() return {} end,
}

namespaces.LrFileUtils = {
	exists = function(path)
		if M.state.existing[path] ~= nil then
			return M.state.existing[path] and 'file' or false
		end
		local handle = io.open(path, 'r')
		if handle then handle:close(); return 'file' end
		return false
	end,
	createAllDirectories = function(path) os.execute('mkdir -p ' .. sh(path)) return true end,
	files = function(folder)
		local handle = io.popen('ls -1 ' .. sh(folder) .. ' 2>/dev/null')
		local names = {}
		if handle then
			for line in handle:lines() do names[#names + 1] = folder .. '/' .. line end
			handle:close()
		end
		local i = 0
		return function() i = i + 1; return names[i] end
	end,
	fileAttributes = function(path)
		local handle = io.open(path, 'rb')
		if not handle then return {} end
		local size = handle:seek('end'); handle:close()
		return { fileSize = size }
	end,
	delete = function(path)
		M.state.deleted = M.state.deleted or {}
		M.state.deleted[#M.state.deleted + 1] = path
		return true
	end,
	readFile = function(path)
		local handle = io.open(path, 'r')
		if not handle then return nil end
		local text = handle:read('*a')
		handle:close()
		-- Reading a file yields in the real SDK; assert we are not inside a gate.
		if M.state.inWriteGate then M.state.yieldInsideWrite = true end
		return text
	end,
}

--- A temp directory of this run's own, made on first use and removed by the
-- next reset or by M.cleanUp (as the suite does with its results file). The
-- machine's temp directory would hand one run the previews an earlier run
-- left, and keep them. Under TMPDIR when a caller sets one (pytest hands its
-- tmp_path), which Lua's own os.tmpname would ignore.
local function tempDir()
	if not M.state.tempDir then
		local base = os.getenv('TMPDIR') or '/tmp'
		local handle = assert(io.popen('mktemp -d ' .. sh(base .. '/lrmock.XXXXXX')))
		-- mktemp prints the path and one newline. The path is TMPDIR's and may
		-- hold newlines of its own; read one line and it comes back cut, naming
		-- TMPDIR's parent, which cleanUp would then remove. Read it whole and
		-- drop only the newline mktemp added.
		local path = (string.gsub(handle:read('*a') or '', '\n$', ''))
		handle:close()
		assert(path ~= '', 'mktemp made no directory under ' .. base)
		M.state.tempDir = path
	end
	return M.state.tempDir
end

-- The host this mock runs on, as distinct from the Lightroom it fakes: Lua
-- spells the directory separator first in package.config.
local HOST_IS_WINDOWS = package.config:sub(1, 1) == '\\'

--- The Windows temp folder of a fake Windows Lightroom: the one the test
-- named in reset's options, if it did. Otherwise, on a Windows host, the
-- real one, TEMP, which is what Lightroom reports there, so a command
-- built for cmd.exe can be run by cmd.exe and the CLI log it names has a
-- folder to land in. Elsewhere a Windows path that exists nowhere: the
-- host's shell could not run the command anyway, and the suites read the
-- line, not the disk.
local function windowsTemp()
	if M.state.windowsTemp then return M.state.windowsTemp end
	if HOST_IS_WINDOWS then return assert(os.getenv('TEMP'), 'TEMP is not set') end
	return 'C:\\Users\\photographer\\AppData\\Local\\Temp'
end

-- Lightroom joins paths with the platform's separator; WIN_ENV picks it.
namespaces.LrPathUtils = {
	child = function(dir, name) return dir .. (WIN_ENV and '\\' or '/') .. name end,
	parent = function(path) return (string.gsub(path, '[/\\][^/\\]+$', '')) end,
	getStandardFilePath = function(which)
		if which == 'temp' then
			if WIN_ENV then return windowsTemp() end
			return tempDir()
		end
		return os.getenv('HOME') or '/tmp'
	end,
}

namespaces.LrPrefs = { prefsForPlugin = function() return M.state.prefs end }

namespaces.LrProgressScope = function(options)
	local scope = { options = options, portions = {}, isDone = false }
	M.state.progressScopes[#M.state.progressScopes + 1] = scope
	scope.setCancelable = function() end
	scope.setPortionComplete = function(_, done, total) scope.portions[#scope.portions + 1] = done / (total or 1) end
	scope.isCanceled = function() return M.state.cancelled end
	scope.done = function() scope.isDone = true end
	return scope
end

namespaces.LrTasks = {
	startAsyncTask = function(func)
		local task = coroutine.create(func)
		M.state.tasks[#M.state.tasks + 1] = task
		resume(task)
	end,
	yield = function() end,
	sleep = function() M.yield() end,
	execute = function(cmd)
		M.state.executed = M.state.executed or {}
		M.state.executed[#M.state.executed + 1] = cmd
		-- state.onExecute plays the executable: given the command, it writes
		-- what the real one would and returns its exit code.
		if M.state.onExecute then return M.state.onExecute(cmd) or 0 end
		return M.state.executeCode or 0
	end,
}

namespaces.LrLogger = function(name)
	return {
		enable = function() end,
		info = function(_, msg) M.state.logLines[#M.state.logLines + 1] = 'INFO ' .. tostring(msg) end,
		warn = function(_, msg) M.state.logLines[#M.state.logLines + 1] = 'WARN ' .. tostring(msg) end,
		error = function(_, msg) M.state.logLines[#M.state.logLines + 1] = 'ERROR ' .. tostring(msg) end,
	}
end

namespaces.LrFunctionContext = {
	callWithContext = function(name, func) return func({}) end,
}

namespaces.LrColor = function() return {} end
namespaces.LrShell = { revealInShell = function() end }
--- The view factory hands back each spec as given, tagged with the kind of
-- view asked for (static_text, group_box, ...), so a dialog's text can be
-- read from the tree the plugin built: children are the array part,
-- attributes the rest. `bind` returns what it was given (a key, or a table
-- with key, bind_to_object and transform), so bindings are inspectable.
namespaces.LrView = {
	osFactory = function()
		return setmetatable({}, { __index = function(_, kind)
			return function(_, spec)
				spec = spec or {}
				spec.kind = kind
				return spec
			end
		end })
	end,
	bind = function(spec) return spec end,
}

--- Install the SDK globals so plugin files can be dofile()'d directly.
-- `options.windows` fakes a Windows Lightroom: WIN_ENV set, MAC_ENV unset,
-- backslash paths. The default is macOS.
function M.install(pluginPath, options)
	options = options or {}
	_G.import = function(name)
		local ns = namespaces[name]
		if ns == nil then error('mock: unknown Lightroom namespace ' .. tostring(name), 2) end
		return ns
	end
	_G._PLUGIN = { path = pluginPath, id = 'net.gregbishop.melampus' }
	_G.WIN_ENV = options.windows and true or nil
	_G.MAC_ENV = (not options.windows) and true or nil
	_G.LOC = function(text) return text end
end

-- ── loading the plugin under the mock ──────────────────────────────────────
-- Shared by the suites that execute the real plugin files, so the plugin's
-- location, its module list and its defaults are spelled once.

--- The plugin folder, relative to this file, so a suite runs from a clone at
--- any path and from any working directory. MELAMPUS_PLUGIN overrides it.
local function pluginPath()
	local here = debug.getinfo(1, 'S').source:match('^@(.*)[/\\]') or '.'
	return here .. '/../Melampus.lrplugin'
end

M.PLUGIN = os.getenv('MELAMPUS_PLUGIN') or pluginPath()

--- The executable beside that plugin on a macOS Lightroom, the path the
--- plugin looks for; a suite tells the mock it is present or absent by name.
M.EXECUTABLE = M.PLUGIN .. '/melampus'

--- Drop the plugin's modules so the next load runs them fresh under the mock.
function M.unloadPlugin()
	for _, name in ipairs({ 'MelampusJson', 'MelampusRules', 'MelampusLog', 'MelampusAnalyze' }) do
		package.loaded[name] = nil
	end
end

--- Load one plugin file fresh, dropping the modules first, and hand back what
--- it returns.
function M.loadPluginFile(name)
	M.unloadPlugin()
	return dofile(M.PLUGIN .. '/' .. name .. '.lua')
end

--- Reset the mock, install it for a plugin folder (this one by default), and
--- load one plugin file fresh under it: the shape every load outside a whole
--- import takes.
function M.loadUnderMock(name, resetOptions, folder, installOptions)
	M.reset(resetOptions)
	M.install(folder or M.PLUGIN, installOptions)
	return M.loadPluginFile(name)
end

--- The plugin's default settings, from MelampusRules.lua loaded fresh, with
--- each table of overrides applied in turn (a nil one is skipped).
function M.defaultPrefs(...)
	local prefs = M.loadPluginFile('MelampusRules').defaultSettings()
	for i = 1, select('#', ...) do
		for k, v in pairs(select(i, ...) or {}) do prefs[k] = v end
	end
	return prefs
end

M.catalog = catalog
return M
