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

function M.reset(options)
	options = options or {}
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
	}
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

function Photo:setRawMetadata(key, value)
	assert(M.state.inWriteGate, 'setRawMetadata outside a write gate')
	self._raw[key] = value
	self._writes[#self._writes + 1] = { key = key, value = value }
end

function Photo:addKeyword(keyword)
	assert(M.state.inWriteGate, 'addKeyword outside a write gate')
	assert(keyword ~= nil, 'addKeyword(nil)')
	for _, existing in ipairs(self._keywords) do
		if existing == keyword then return end
	end
	self._keywords[#self._keywords + 1] = keyword
end

function Photo:getPropertyForPlugin(_, field) return self._plugin[field] end

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
	presentModalDialog = function() return 'ok' end,
}

namespaces.LrFileUtils = {
	exists = function(path)
		local handle = io.open(path, 'r')
		if handle then handle:close(); return 'file' end
		return false
	end,
	createAllDirectories = function(path) os.execute('mkdir -p ' .. path) return true end,
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

namespaces.LrPathUtils = {
	child = function(dir, name) return dir .. '/' .. name end,
	parent = function(path) return (string.gsub(path, '/[^/]+$', '')) end,
	getStandardFilePath = function(which)
		if which == 'temp' then return '/tmp' end
		return os.getenv('HOME') or '/tmp'
	end,
}

namespaces.LrPrefs = { prefsForPlugin = function() return M.state.prefs end }

namespaces.LrProgressScope = function(options)
	return {
		setCancelable = function() end,
		setPortionComplete = function() end,
		isCanceled = function() return M.state.cancelled end,
		done = function() end,
	}
end

namespaces.LrTasks = {
	startAsyncTask = function(func) func() end,
	yield = function() end,
	sleep = function() end,
	execute = function(cmd)
		M.state.executed = M.state.executed or {}
		M.state.executed[#M.state.executed + 1] = cmd
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
namespaces.LrView = {
	osFactory = function()
		return setmetatable({}, { __index = function() return function() return {} end end })
	end,
	bind = function(key) return key end,
}

--- Install the SDK globals so plugin files can be dofile()'d directly.
function M.install(pluginPath)
	_G.import = function(name)
		local ns = namespaces[name]
		if ns == nil then error('mock: unknown Lightroom namespace ' .. tostring(name), 2) end
		return ns
	end
	_G._PLUGIN = { path = pluginPath, id = 'net.gregbishop.melampus' }
	_G.LOC = function(text) return text end
end

M.catalog = catalog
return M
