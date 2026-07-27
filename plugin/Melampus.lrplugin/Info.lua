--[[
Melampus — plugin manifest and custom metadata schema.

Review-only build. It reads identification results produced by the Python side
from a JSON file on disk and writes them into the catalog, so you review inside
Lightroom's own grid and loupe rather than a separate window. There is no HTTP
service and no bundled binary yet; those arrive with Stage 3.

Menu surface is deliberately four items (CLAUDE.md §5.4.2). Everything else —
force reprocess, clearing state, diagnostics — lives inside Settings.

A note on the metadata schema below. The SDK supports exactly three field types:
string, enum and URL. Enum values are fixed here at definition time and cannot
vary per photo, which is why the ranked alternates are shown as a read-only
string rather than a selectable list: each photo's alternates differ, and an enum
cannot be populated per photo. The verdict field is an enum because its values
genuinely are fixed.
--]]

return {
	LrSdkVersion = 12.0,
	LrSdkMinimumVersion = 10.0,

	LrToolkitIdentifier = 'net.gregbishop.melampus',
	LrPluginName = 'Melampus',
	LrPluginInfoUrl = 'https://github.com/',

	LrInitPlugin = 'MelampusInit.lua',

	LrMetadataProvider = 'MelampusMetadata.lua',
	LrMetadataTagsetFactory = 'MelampusTagset.lua',

	LrExportMenuItems = {},

	-- Four items, no more. A cluttered plugin menu makes a tool feel unfinished.
	LrLibraryMenuItems = {
		{
			title = 'Melampus: Identify Selected Photos…',
			file = 'MelampusImport.lua',
			enabledWhen = 'photosAvailable',
		},
		{
			title = 'Melampus: Set Up Review Collections',
			file = 'MelampusReviewQueue.lua',
		},
		{
			title = 'Melampus: Settings…',
			file = 'MelampusSettings.lua',
		},
		{
			title = 'Melampus: Save My Corrections…',
			file = 'MelampusLogCorrections.lua',
		},
	},

	VERSION = { major = 0, minor = 1, revision = 0, build = 1 },
}
