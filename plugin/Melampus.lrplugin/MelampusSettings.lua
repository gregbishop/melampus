--[[ Settings. Everything beyond the four menu items lives here (§5.4.2):
     the results file, the safety switches, the gates, and maintenance. ]]
local LrDialogs = import 'LrDialogs'
local LrFunctionContext = import 'LrFunctionContext'
local LrPrefs = import 'LrPrefs'
local LrTasks = import 'LrTasks'
local LrView = import 'LrView'

local Rules = require 'MelampusRules'

LrTasks.startAsyncTask(function()
	LrFunctionContext.callWithContext('melampusSettings', function(context)
		local prefs = LrPrefs.prefsForPlugin()
		local f = LrView.osFactory()
		local bind = LrView.bind

		local contents = f:column {
			bind_to_object = prefs,
			spacing = f:control_spacing(),

			f:group_box {
				title = 'Results file',
				fill_horizontal = 1,
				f:row {
					f:edit_field { value = bind 'resultsPath', width_in_chars = 42,
						immediate = true, tooltip = 'JSON written by melampus-id --json-out' },
					f:push_button {
						title = 'Choose…',
						action = function()
							local chosen = LrDialogs.runOpenPanel {
								title = 'Select Melampus results JSON',
								canChooseFiles = true, canChooseDirectories = false,
								allowsMultipleSelection = false,
							}
							if chosen and chosen[1] then prefs.resultsPath = chosen[1] end
						end,
					},
				},
			},

			f:group_box {
				title = 'Safety',
				fill_horizontal = 1,
				f:checkbox { title = 'Dry run — report changes without writing',
					value = bind 'dryRun' },
				f:static_text {
					title = 'Leave this on for the first run against any catalog.',
					text_color = import('LrColor')(0.4, 0.4, 0.4),
				},
				f:checkbox { title = 'Force — reprocess photos already done',
					value = bind 'force' },
			},

			f:group_box {
				title = 'What to write',
				fill_horizontal = 1,
				f:checkbox { title = 'Hierarchical keywords', value = bind 'writeKeywords' },
				f:checkbox { title = 'Custom metadata fields', value = bind 'writeMetadata' },
				f:checkbox { title = 'Star rating (needs quality scores — Stage 2)',
					value = bind 'writeRating' },
				f:checkbox { title = 'Colour label', value = bind 'writeLabel' },
				f:checkbox { title = 'Pick flags', value = bind 'writeFlags' },
				f:checkbox { title = 'Auto-reject poor frames (off by default)',
					value = bind 'autoReject' },
			},

			f:group_box {
				title = 'Overwrite permission — off means write only where empty',
				fill_horizontal = 1,
				f:checkbox { title = 'Overwrite existing ratings', value = bind 'overwriteRating' },
				f:checkbox { title = 'Overwrite existing labels', value = bind 'overwriteLabel' },
				f:checkbox { title = 'Overwrite existing flags', value = bind 'overwriteFlags' },
			},

			f:group_box {
				title = 'Auto-tag gates',
				fill_horizontal = 1,
				f:row {
					f:static_text { title = 'Minimum confidence:' },
					f:edit_field { value = bind 'minConfidence', width_in_chars = 5,
						min = 0, max = 1, precision = 2 },
					f:static_text { title = 'Minimum burst agreement:' },
					f:edit_field { value = bind 'minBurstAgreement', width_in_chars = 5,
						min = 0, max = 1, precision = 2 },
				},
				f:static_text {
					title = 'Below either gate a review keyword is written instead of a species.',
					text_color = import('LrColor')(0.4, 0.4, 0.4),
				},
			},

			f:group_box {
				title = 'Maintenance',
				fill_horizontal = 1,
				f:push_button {
					title = 'Restore safe defaults',
					action = function()
						for key, value in pairs(Rules.defaultSettings()) do prefs[key] = value end
						LrDialogs.message('Melampus', 'Settings restored to safe defaults.', 'info')
					end,
				},
			},
		}

		LrDialogs.presentModalDialog {
			title = 'Melampus Settings',
			contents = contents,
			actionVerb = 'Done',
			cancelVerb = '< exclude >',
		}
	end)
end)
