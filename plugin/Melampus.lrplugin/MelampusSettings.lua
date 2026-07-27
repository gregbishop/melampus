--[[ Settings. Everything beyond the four menu items lives here (§5.4.2):
     the results file, the safety switches, the gates, and maintenance. ]]
local LrDialogs = import 'LrDialogs'
local LrFunctionContext = import 'LrFunctionContext'
local LrPrefs = import 'LrPrefs'
local LrTasks = import 'LrTasks'
local LrView = import 'LrView'

local Log = require 'MelampusLog'
local Rules = require 'MelampusRules'

LrTasks.startAsyncTask(function()
	LrFunctionContext.callWithContext('melampusSettings', function(context)
		local prefs = LrPrefs.prefsForPlugin()
		local f = LrView.osFactory()
		local bind = LrView.bind

		local contents = f:column {
			bind_to_object = prefs,
			spacing = f:control_spacing(),

			f:static_text {
				title = 'Melampus reads species identifications that were worked out on\n'
					.. 'your Mac, and puts them onto your photos as keywords.',
				height_in_lines = 2,
			},

			f:group_box {
				title = 'Step 1 — where the identifications are',
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
				title = 'Step 2 — safety',
				fill_horizontal = 1,
				f:checkbox {
					title = 'Preview only — show me what would change, change nothing',
					value = bind 'dryRun' },
				f:static_text {
					title = 'Keep this ticked until you have seen a preview you are happy with.\n'
						.. 'Untick it when you want the changes actually applied.',
					height_in_lines = 2,
					text_color = import('LrColor')(0.4, 0.4, 0.4),
				},
				f:checkbox { title = 'Redo photos I have already done',
					value = bind 'force' },
			},

			f:group_box {
				title = 'What to add to photos',
				fill_horizontal = 1,
				f:checkbox { title = 'Keywords (species, taxon, review status)', value = bind 'writeKeywords' },
				f:checkbox { title = 'Melampus panel details (confidence, alternates)', value = bind 'writeMetadata' },
				f:checkbox { title = 'Star rating (needs quality scores — Stage 2)',
					value = bind 'writeRating' },
				f:checkbox { title = 'Colour label', value = bind 'writeLabel' },
				f:checkbox { title = 'Pick flags', value = bind 'writeFlags' },
				f:checkbox { title = 'Auto-reject poor frames (off by default)',
					value = bind 'autoReject' },
			},

			f:group_box {
				title = 'Never touch things I set myself (leave these unticked)',
				fill_horizontal = 1,
				f:checkbox { title = 'Overwrite existing ratings', value = bind 'overwriteRating' },
				f:checkbox { title = 'Overwrite existing labels', value = bind 'overwriteLabel' },
				f:checkbox { title = 'Overwrite existing flags', value = bind 'overwriteFlags' },
			},

			f:group_box {
				title = 'How sure must it be before naming a species',
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
					title = 'Not sure enough? It adds "Needs ID" instead of guessing a species.',
					text_color = import('LrColor')(0.4, 0.4, 0.4),
				},
			},

			f:group_box {
				title = 'Maintenance',
				fill_horizontal = 1,
				f:static_text { title = 'Log: ' .. Log.path(), width_in_chars = 50 },
				f:row {
					f:push_button {
						title = 'Show log in Finder',
						action = function()
							import('LrShell').revealInShell(Log.path())
						end,
					},
					f:push_button {
						title = 'Check results file',
						action = function()
							local LrFileUtils = import 'LrFileUtils'
							local path = prefs.resultsPath
							if not path or path == '' then
								LrDialogs.message('Melampus', 'No results file set.', 'warning')
								return
							end
							if not LrFileUtils.exists(path) then
								LrDialogs.message('Melampus', 'Not found:\n' .. path, 'critical')
								return
							end
							local text = LrFileUtils.readFile(path)
							local Json = require 'MelampusJson'
							local data, err = Json.decode(text)
							if not data then
								LrDialogs.message('Melampus', 'Parse failed:\n' .. tostring(err), 'critical')
								return
							end
							local names = {}
							for i = 1, math.min(3, #data) do names[#names + 1] = data[i].file end
							LrDialogs.message('Melampus',
								string.format('OK — %d records.\n\nFirst filenames:\n  %s\n\n'
									.. 'Your catalog photos must share these basenames.',
									#data, table.concat(names, '\n  ')), 'info')
						end,
					},
				},
				f:push_button {
					title = 'Remove all Melampus keywords…',
					action = function()
						local LrApplication = import 'LrApplication'
						local LrTasks = import 'LrTasks'
						local choice = LrDialogs.confirm('Remove Melampus keywords',
							'This deletes the Melampus keyword tree and every stray '
							.. 'top-level keyword it created (Species, Taxon, Confidence, '
							.. 'Review, Notable, Needs ID).\n\nYour own keywords are '
							.. 'untouched. Photos keep everything else.',
							'Remove them', 'Cancel')
						if choice ~= 'ok' then return end
						LrTasks.startAsyncTask(function()
							local catalog = LrApplication.activeCatalog()
							local strays = {
								['melampus'] = true, ['species'] = true, ['taxon'] = true,
								['confidence'] = true, ['review'] = true,
								['notable'] = true, ['needs id'] = true,
								['out of range'] = true,
							}
							local removed = 0
							catalog:withWriteAccessDo('Melampus: remove keywords', function()
								for _, keyword in ipairs(catalog:getKeywords() or {}) do
									if strays[string.lower(keyword:getName())] then
										catalog:deleteKeyword(keyword)
										removed = removed + 1
									end
								end
							end, { timeout = 60 })
							LrDialogs.message('Melampus',
								string.format('Removed %d top-level keyword trees.', removed), 'info')
						end)
					end,
				},
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
