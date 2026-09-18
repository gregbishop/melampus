--[[ The release zip's plugin folder loads (card #402, the part that needs no
     Lightroom): Info.lua parses under a plain interpreter, and every file it
     names, plus every module those files require, is in the zip listing.
     MELAMPUS_PLUGIN_DIR is the unpacked Melampus.lrplugin folder;
     MELAMPUS_ZIP_LISTING is the zip's entries, one per line. ]]
local t = require('harness')

local dir = assert(os.getenv('MELAMPUS_PLUGIN_DIR'), 'MELAMPUS_PLUGIN_DIR is not set')
local listing = {}
for entry in (os.getenv('MELAMPUS_ZIP_LISTING') or ''):gmatch('[^\n]+') do
	listing[#listing + 1] = entry
end
local folder = 'Melampus.lrplugin'

local function read(name)
	local handle = assert(io.open(dir .. '/' .. name, 'rb'), 'cannot read ' .. name)
	local text = handle:read('*a')
	handle:close()
	return text
end

local function namedFiles(info)
	local names = { info.LrInitPlugin, info.LrMetadataProvider, info.LrMetadataTagsetFactory }
	for _, item in ipairs(info.LrLibraryMenuItems or {}) do names[#names + 1] = item.file end
	for _, item in ipairs(info.LrExportMenuItems or {}) do names[#names + 1] = item.file end
	return names
end

t.test('Info.lua parses to the plugin manifest', function()
	local info = dofile(dir .. '/Info.lua')
	t.equals(type(info), 'table')
	t.equals(info.LrToolkitIdentifier, 'net.gregbishop.melampus')
	t.equals(info.LrPluginName, 'Melampus')
	t.isNotNil(info.LrSdkVersion)
end)

t.test('the listing holds Info.lua at the root of one plugin folder', function()
	t.isTrue(#listing > 0, 'MELAMPUS_ZIP_LISTING is empty')
	t.contains(listing, folder .. '/Info.lua')
	for _, entry in ipairs(listing) do
		t.isTrue(entry:sub(1, #folder + 1) == folder .. '/', entry .. ' is outside ' .. folder)
	end
end)

t.test('every file Info.lua names is in the zip', function()
	local names = namedFiles(dofile(dir .. '/Info.lua'))
	t.isTrue(#names >= 4, 'Info.lua names fewer files than the four menu items')
	for _, name in ipairs(names) do
		t.contains(listing, folder .. '/' .. name, 'named in Info.lua but not in the zip')
	end
end)

t.test('every module the plugin requires is in the zip', function()
	local seen, queue = {}, namedFiles(dofile(dir .. '/Info.lua'))
	local required = 0
	while #queue > 0 do
		local name = table.remove(queue)
		if not seen[name] then
			seen[name] = true
			for module in read(name):gmatch("require%s*%(?%s*['\"]([%w_]+)['\"]") do
				required = required + 1
				t.contains(listing, folder .. '/' .. module .. '.lua', 'required by ' .. name .. ' but not in the zip')
				queue[#queue + 1] = module .. '.lua'
			end
		end
	end
	t.isTrue(required > 0, 'no require found: the scan is broken')
end)

return t.summary()
