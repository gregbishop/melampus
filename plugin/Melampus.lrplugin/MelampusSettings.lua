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
				f:row {
					f:static_text { title = 'Update the grid every' },
					f:popup_menu {
						value = bind 'analyzeBatchSize',
						items = {
							{ title = '10 photos  (updates often, slightly slower)', value = 10 },
							{ title = '25 photos  (recommended)', value = 25 },
							{ title = '50 photos', value = 50 },
							{ title = '100 photos  (fewest interruptions)', value = 100 },
						},
					},
				},
				f:static_text {
					title = 'Analysis runs in batches. Each batch is written to your catalog\n'
						.. 'before the next starts, so stars and keywords appear as it goes\n'
						.. 'instead of all at the end. At roughly 7 seconds a photo, 25 is\n'
						.. 'about three minutes between updates. Cancelling keeps every\n'
						.. 'batch that finished.',
					height_in_lines = 5,
					text_color = import('LrColor')(0.4, 0.4, 0.4),
				},
				f:checkbox { title = 'Keep the working previews after analysing',
					value = bind 'keepPreviews' },
				f:static_text {
					title = 'Previews are temporary JPEGs Melampus makes to look at your\n'
						.. 'photos. They never enter your catalog and are deleted when\n'
						.. 'analysis succeeds. About 78 KB each; keep them only to debug.',
					height_in_lines = 3,
					text_color = import('LrColor')(0.4, 0.4, 0.4),
				},
			},

			f:group_box {
				title = 'What to add to photos',
				fill_horizontal = 1,
				f:checkbox { title = 'Keywords (species, taxon, review status)', value = bind 'writeKeywords' },
				f:checkbox { title = 'Melampus panel details (confidence, alternates)', value = bind 'writeMetadata' },
				f:checkbox { title = 'Behaviour keywords (in-flight, feeding, wading…)',
					value = bind 'writeBehaviour' },
				f:checkbox { title = 'Age and sex keywords (adult male, juvenile)',
					value = bind 'writeAgeSex' },
				f:checkbox { title = 'Star rating (needs quality scores — Stage 2)',
					value = bind 'writeRating' },
				f:checkbox { title = 'Colour label — green identified, yellow needs a look, red out of range',
					value = bind 'writeLabel' },
				f:static_text {
					title = 'Red is the interesting pile: a species named that does not\n'
						.. 'occur near Merritt Island. Either the model is wrong, or you\n'
						.. 'photographed something genuinely unusual.',
					height_in_lines = 3,
					text_color = import('LrColor')(0.4, 0.4, 0.4),
				},
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
					title = 'Clean up Melampus keywords…',
					action = function()
						local LrApplication = import 'LrApplication'
						local LrTasks = import 'LrTasks'
						LrTasks.startAsyncTask(function()
							local catalog = LrApplication.activeCatalog()

							-- Only the structural keywords Melampus creates. Species
							-- names are deliberately NOT touched: under the flat
							-- scheme they are the correct output, and they may also
							-- be keywords you added yourself.
							local structural = {
								['melampus'] = true, ['species'] = true, ['taxon'] = true,
								['confidence'] = true, ['review'] = true, ['notable'] = true,
								['high'] = true, ['medium'] = true, ['low'] = true,
							}

							local doomed, lines = {}, {}
							for _, keyword in ipairs(catalog:getKeywords() or {}) do
								local name = keyword:getName()
								if structural[string.lower(name)] then
									local n = #(keyword:getPhotos() or {})
									local kids = #(keyword:getChildren() or {})
									doomed[#doomed + 1] = keyword
									lines[#lines + 1] = string.format(
										'   %s  (%d photos, %d sub-keywords)', name, n, kids)
								end
							end

							if #doomed == 0 then
								LrDialogs.message('Melampus',
									'Nothing to clean up. No Melampus structural keywords found.',
									'info')
								return
							end

							-- Show exactly what goes, before anything goes.
							local choice = LrDialogs.confirm('Clean up Melampus keywords',
								'These keywords and everything nested under them will be '
								.. 'deleted:\n\n' .. table.concat(lines, '\n')
								.. '\n\nSpecies names are NOT touched — under the flat '
								.. 'scheme those are the real keywords.\n\n'
								.. 'Your photos keep every other keyword. This cannot be undone '
								.. 'from here, though Lightroom\'s Undo will reverse it.',
								'Delete these', 'Cancel')
							if choice ~= 'ok' then return end

							local removed = 0
							catalog:withWriteAccessDo('Melampus: clean up keywords', function()
								for _, keyword in ipairs(doomed) do
									catalog:deleteKeyword(keyword)
									removed = removed + 1
								end
							end, { timeout = 60 })
							LrDialogs.message('Melampus',
								string.format('Removed %d keyword trees.', removed), 'info')
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
