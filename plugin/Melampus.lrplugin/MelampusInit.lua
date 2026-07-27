--[[ Plugin init. Deliberately minimal: no catalog access, no network, no work
     that could delay Lightroom's startup. Preferences are seeded to the safe
     defaults so a fresh install cannot write to a catalog on its first run. ]]
local LrPrefs = import 'LrPrefs'
local Rules = require 'MelampusRules'

local prefs = LrPrefs.prefsForPlugin()
for key, value in pairs(Rules.defaultSettings()) do
	if prefs[key] == nil then prefs[key] = value end
end
if prefs.resultsPath == nil then prefs.resultsPath = '' end
