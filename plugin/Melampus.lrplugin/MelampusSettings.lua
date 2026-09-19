--[[ Settings. Everything beyond the four menu items lives here (§5.4.2):
     the results file, the safety switches, the gates, and maintenance. ]]
local LrBinding = import 'LrBinding'
local LrColor = import 'LrColor'
local LrDialogs = import 'LrDialogs'
local LrFunctionContext = import 'LrFunctionContext'
local LrHttp = import 'LrHttp'
local LrPasswords = import 'LrPasswords'
local LrPrefs = import 'LrPrefs'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'
local LrView = import 'LrView'

local Analyze = require 'MelampusAnalyze'
local Log = require 'MelampusLog'
local Rules = require 'MelampusRules'

-- Lightroom's Lua 5.1 has unpack; the local interpreter the tests use has
-- table.unpack.
local unpack = unpack or table.unpack

local function lineCount(text)
	local _, newlines = string.gsub(text, '\n', '')
	return newlines + 1
end

LrTasks.startAsyncTask(function()
	LrFunctionContext.callWithContext('melampusSettings', function(context)
		local prefs = LrPrefs.prefsForPlugin()
		local f = LrView.osFactory()
		local bind = LrView.bind
		-- The explanatory text under a control, everywhere in the dialog.
		local grey = LrColor(0.4, 0.4, 0.4)

		-- Which engines can run here is the executable's verdict (card #404),
		-- asked once, now, as the dialog opens. Without the executable nothing
		-- is greyed and the note says what is missing.
		local verdicts, problem = Analyze.detectEngines()
		local engineItems, engineNote = Rules.engineItems(verdicts, problem)

		-- The picker, the reasons for whatever is greyed, and one link: Ollama's
		-- alone, where to install it, taken from the executable's reason when
		-- Ollama is unavailable.
		local engineViews = {
			f:popup_menu { value = bind 'engine', items = engineItems },
		}
		if engineNote ~= '' then
			engineViews[#engineViews + 1] = f:static_text {
				title = engineNote, height_in_lines = lineCount(engineNote), text_color = grey,
			}
		end
		for _, item in ipairs(engineItems) do
			if not item.enabled and item.link then
				local link = item.link
				engineViews[#engineViews + 1] = f:static_text {
					title = link,
					text_color = LrColor(0.1, 0.3, 0.8),
					mouse_down = function() LrHttp.openUrlInBrowser(link) end,
				}
			end
		end

		-- A cloud engine's API key. It lives in LrPasswords (the OS keychain on
		-- macOS), never in the preferences, so the fields bind to their own
		-- table: read from the store as the dialog opens, written back when it
		-- closes. Each field shows only while its engine is picked: the row's
		-- visibility names the preferences as its target, since the field
		-- beside it is where the target changes from the column's to the keys.
		local keys, keysAtOpen = LrBinding.makePropertyTable(context), {}
		for _, engine in ipairs(Rules.ENGINES) do
			local variable = Rules.keyVariable(engine)
			if variable then
				keys[variable] = LrPasswords.retrieve(variable) or ''
				keysAtOpen[variable] = keys[variable]
				engineViews[#engineViews + 1] = f:row {
					visible = bind {
						key = 'engine', bind_to_object = prefs,
						transform = function(value) return value == engine end,
					},
					f:static_text { title = 'API key:' },
					f:password_field {
						bind_to_object = keys, value = bind(variable),
						width_in_chars = 42, immediate = true,
						tooltip = 'Kept in the system\'s secure store, not in a file',
					},
				}
			end
		end

		-- The MLX model (card #408). While it is absent, a button downloads
		-- it, named with the model and its size; while it downloads, the
		-- bytes so far (Lightroom's own progress bar carries the portion) and
		-- Cancel; once present, Installed and Remove. The status is asked
		-- once, now, and only where mlx can run at all; the row shows when
		-- the picked engine is mlx, or the unset preference resolves to it.
		local mlxHere = false
		for _, verdict in ipairs(type(verdicts) == 'table' and verdicts or {}) do
			if type(verdict) == 'table' and verdict.engine == 'mlx' and verdict.available == true then mlxHere = true end
		end
		local status, statusProblem
		if mlxHere then status, statusProblem = Analyze.modelStatus() end
		if statusProblem then
			engineViews[#engineViews + 1] = f:static_text {
				title = statusProblem, height_in_lines = lineCount(statusProblem), text_color = grey,
			}
		end
		if status then
			local model = LrBinding.makePropertyTable(context)
			model.phase = status.installed == true and 'installed' or 'absent'
			model.progress = ''
			local function inPhase(name)
				return bind { key = 'phase', object = model, transform = function(value) return value == name end }
			end
			local download = nil

			local function startDownload()
				-- Not tied to the dialog's context: the download outlives a
				-- closed Settings dialog (nothing can kill the executable
				-- anyway), and the scope ends when the command exits.
				local scope = LrProgressScope { title = 'Downloading ' .. tostring(status.repo) }
				scope:setCancelable(true)
				model.phase, model.progress = 'downloading', 'Starting…'
				local handle, err = Analyze.downloadModel(status.cancel_path,
					function(update)
						if update.state == 'progress' then
							local text, portion = Rules.downloadProgress(update)
							model.progress = text
							scope:setPortionComplete(portion, 1)
						end
						-- Lightroom's own cancel, on its progress bar, cancels too.
						if scope:isCanceled() and download then download.cancel() end
					end,
					function(code, update, tail)
						scope:done()
						download = nil
						if code == 0 and update and update.state == 'done' then
							model.phase = 'installed'
							return
						end
						model.phase = 'absent'
						if code ~= 4 then
							LrDialogs.message('Melampus', 'The model download failed (exit ' .. tostring(code)
								.. ').\n\n' .. tostring(tail), 'critical')
						end
					end)
				download = handle
				if not handle then
					scope:done()
					model.phase = 'absent'
					LrDialogs.message('Melampus', tostring(err), 'critical')
				end
			end

			local function removeModel()
				LrTasks.startAsyncTask(function()
					local ok, message = Analyze.removeModel()
					if ok then model.phase = 'absent' else LrDialogs.message('Melampus', message, 'critical') end
				end)
			end

			engineViews[#engineViews + 1] = f:row {
				visible = bind {
					key = 'engine', object = prefs,
					transform = function(value) return Rules.resolvedEngine(value, verdicts) == 'mlx' end,
				},
				bind_to_object = model,
				f:push_button { title = Rules.downloadTitle(status), visible = inPhase('absent'), action = startDownload },
				f:push_button { title = 'Installed', enabled = false, visible = inPhase('installed') },
				f:push_button { title = 'Remove', visible = inPhase('installed'), action = removeModel },
				f:static_text { title = bind 'progress', visible = inPhase('downloading'), width_in_chars = 24 },
				f:push_button {
					title = 'Cancel', visible = inPhase('downloading'),
					action = function() if download then download.cancel() end end,
				},
			}
		end

		local contents = f:column {
			bind_to_object = prefs,
			spacing = f:control_spacing(),

			f:static_text {
				title = 'Melampus analyses the photos you select with the program in its\n'
					.. 'own plugin folder, and puts the species it finds on them as keywords.',
				height_in_lines = 2,
			},

			f:group_box {
				title = 'Optional — results produced elsewhere',
				fill_horizontal = 1,
				f:row {
					f:edit_field { value = bind 'resultsPath', width_in_chars = 42,
						immediate = true,
						tooltip = 'JSON written by melampus-id --plugin-out. Leave empty to analyse here.' },
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
				title = 'Where identification runs',
				fill_horizontal = 1,
				unpack(engineViews),
			},

			f:group_box {
				title = 'What kind of photos are these?',
				fill_horizontal = 1,
				f:popup_menu {
					value = bind 'profile',
					items = {
						{ title = 'Wildlife — identify the species', value = 'wildlife' },
						{ title = 'Sport — identify the activity and the action', value = 'sport' },
					},
				},
				f:static_text {
					title = 'Set this before analysing. Wildlife asks what organism is in\n'
						.. 'the frame and checks it against range data. Sport asks what is\n'
						.. 'happening — the movement, the equipment, the moment.',
					height_in_lines = 3,
					text_color = grey,
				},
			},

			f:group_box {
				title = 'Safety',
				fill_horizontal = 1,
				f:checkbox {
					title = 'Preview only — show me what would change, change nothing',
					value = bind 'dryRun' },
				f:static_text {
					title = 'Keep this ticked until you have seen a preview you are happy with.\n'
						.. 'Untick it when you want the changes actually applied.',
					height_in_lines = 2,
					text_color = grey,
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
					text_color = grey,
				},
				f:checkbox { title = 'Keep the working previews after analysing',
					value = bind 'keepPreviews' },
				f:static_text {
					title = 'Previews are temporary JPEGs Melampus makes to look at your\n'
						.. 'photos. They never enter your catalog and are deleted when\n'
						.. 'analysis succeeds. About 78 KB each; keep them only to debug.',
					height_in_lines = 3,
					text_color = grey,
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
					text_color = grey,
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
					text_color = grey,
				},
			},

			f:group_box {
				title = 'Maintenance',
				fill_horizontal = 1,
				f:static_text { title = 'Log: ' .. Log.path(), width_in_chars = 50 },
				f:row {
					f:push_button {
						title = 'Show log file',
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

		-- The dialog has no Cancel, so what the fields hold is what the user
		-- wants kept: a changed field is stored, an emptied one forgets the
		-- key, an untouched one leaves the store alone.
		for variable, before in pairs(keysAtOpen) do
			local after = keys[variable] or ''
			if after ~= before then LrPasswords.store(variable, after) end
		end
	end)
end)
