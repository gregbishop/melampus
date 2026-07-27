--[[
Import identifications into the catalog.

The ordering here is the whole point, and it follows CLAUDE.md §5.3: every piece
of reading, parsing and decision-making finishes *before* a write transaction
opens. Yielding inside withWriteAccessDo is what produces "yielding is not
allowed" errors, and file I/O yields.

So: read the file, decode it, read existing photo state, build every plan — then
open one transaction per chunk and apply. Nothing async happens inside a gate.

Dry run is the default. The first run against any catalog reports what it would
do and writes nothing.
--]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrFileUtils = import 'LrFileUtils'
local LrFunctionContext = import 'LrFunctionContext'
local LrPathUtils = import 'LrPathUtils'
local LrPrefs = import 'LrPrefs'
local LrProgressScope = import 'LrProgressScope'
local LrTasks = import 'LrTasks'

local Analyze = require 'MelampusAnalyze'
local Json = require 'MelampusJson'
local Log = require 'MelampusLog'
local Rules = require 'MelampusRules'

local PLUGIN_ID = 'net.gregbishop.melampus'
local CHUNK = 100 -- photos per write transaction, so a crash loses little

--- Look for the results file in the obvious places before giving up.
-- The plugin lives at <repo>/plugin/Melampus.lrplugin, so the file the Python
-- side writes is normally two levels up. Finding it automatically removes the
-- single configuration step that made the plugin look broken.
local function discoverResults()
	local candidates = {}
	local pluginDir = _PLUGIN and _PLUGIN.path
	if pluginDir then
		local repo = LrPathUtils.parent(LrPathUtils.parent(pluginDir))
		if repo then
			candidates[#candidates + 1] = LrPathUtils.child(repo, 'plugin_results.json')
			candidates[#candidates + 1] = LrPathUtils.child(repo, 'stage1_full_results.json')
		end
		candidates[#candidates + 1] = LrPathUtils.child(pluginDir, 'plugin_results.json')
	end
	for _, candidate in ipairs(candidates) do
		if LrFileUtils.exists(candidate) then
			Log.info('auto-discovered results at ' .. candidate)
			return candidate
		end
	end
	return nil
end

local function readResults(path)
	if not path or path == '' then
		path = discoverResults()
	end
	if not path or path == '' then
		return nil, 'FIRST_RUN'
	end
	if not LrFileUtils.exists(path) then
		return nil, 'Results file not found:\n' .. path
	end
	local text = LrFileUtils.readFile(path)
	if not text or text == '' then
		return nil, 'Results file is empty:\n' .. path
	end
	local data, err = Json.decode(text)
	if not data then
		return nil, 'Could not parse the results file.\n\n' .. tostring(err)
	end
	if type(data) ~= 'table' then
		return nil, 'Results file did not contain a list of records.'
	end
	return data, nil, path
end

--- Flatten one Python-side record into the shape Rules expects.
local function normalise(record)
	local ident = record.identification
	local top, alternates = nil, {}
	if type(ident) == 'table' and type(ident.candidates) == 'table' then
		for index, cand in ipairs(ident.candidates) do
			if index == 1 then
				top = cand
			else
				alternates[#alternates + 1] = cand.common_name
			end
		end
	end
	local abstain = false
	if type(ident) == 'table' and ident.abstain == true then abstain = true end

	return {
		file = record.file,
		status = record.status,
		species = top and top.common_name or nil,
		scientificName = top and top.scientific_name or nil,
		confidence = top and top.confidence or nil,
		alternates = #alternates > 0 and table.concat(alternates, ', ') or nil,
		taxon = type(ident) == 'table' and ident.taxon or nil,
		abstain = abstain,
		-- These three are supplied by the encounter-aware exporter. Absent is
		-- handled: Rules treats a missing agreement as "unknown", not "failed".
		burstAgreement = record.burst_agreement,
		rangeFlag = record.range_flag == true,
		encounter = record.encounter,
		quality = record.quality,
		qualityRank = record.quality_rank,
		encounterFrames = record.encounter_frames,
		model = record.model,
	}
end

--- Existing catalog state for a photo, as a plain table Rules can reason about.
local function photoState(photo)
	local keywords = {}
	for _, keyword in ipairs(photo:getRawMetadata('keywords') or {}) do
		local parts, node = {}, keyword
		while node do
			table.insert(parts, 1, node:getName())
			node = node:getParent()
		end
		keywords[#keywords + 1] = table.concat(parts, ' > ')
	end

	local stored = {}
	for _, field in ipairs({ 'species', 'model', 'schemaVersion', 'verdict' }) do
		stored[field] = photo:getPropertyForPlugin(_PLUGIN, field)
	end

	return {
		rating = photo:getRawMetadata('rating'),
		pickStatus = photo:getRawMetadata('pickStatus'),
		colorNameForLabel = photo:getRawMetadata('colorNameForLabel'),
		keywords = keywords,
		melampus = stored,
	}
end

--- Find or create a keyword from a "A > B > C" path. Must run inside a write gate.
local function keywordFromPath(catalog, path)
	local parent = nil
	for segment in string.gmatch(path, '[^>]+') do
		local name = string.gsub(segment, '^%s*(.-)%s*$', '%1')
		if name ~= '' then
			-- Prefer an existing child of the current parent. createKeyword with
			-- returnExisting can return nil when the name collides elsewhere in
			-- the tree, and a nil parent silently dumps every remaining segment
			-- at the root — which is how "Species" and "Taxon" ended up as
			-- top-level keywords instead of under Melampus.
			local found = nil
			local siblings = parent and parent:getChildren() or catalog:getKeywords()
			for _, candidate in ipairs(siblings or {}) do
				if candidate:getName() == name then
					found = candidate
					break
				end
			end

			if found == nil then
				found = catalog:createKeyword(name, {}, false, parent, true)
			end
			if found == nil then
				-- Still nothing: abandon this keyword rather than reparent the
				-- rest of the path to the root and pollute the keyword list.
				Log.warn('could not create or find keyword "' .. name
					.. '" under ' .. (parent and parent:getName() or 'root')
					.. '; skipping "' .. path .. '"')
				return nil
			end
			parent = found
		end
	end
	return parent
end

--- What this plan will actually claim about the photo, in words.
-- "3 keywords, 11 fields" tells you nothing about whether the answer is right.
-- The species name and how sure it is are the only things worth previewing.
--- Apply a list of {photo, plan} inside one write transaction.
-- Shared by the batch path and the bulk path so incremental and one-shot
-- application cannot drift apart.
local applyPlans

--- What this plan will actually claim about the photo, in words.
local function describe(plan, source)
	if plan.metadata and plan.metadata.species then
		local line = plan.metadata.species
		if plan.metadata.confidence then
			line = line .. ' (' .. tostring(math.floor(tonumber(plan.metadata.confidence) * 100 + 0.5)) .. '%)'
		end
		if plan.metadata.rangeFlag == 'out-of-range' then
			line = line .. '  [not found near here]'
		end
		return line
	end
	if source and source.abstain then return 'Needs ID — could not tell' end
	if source and source.rangeFlag then return 'Needs ID — species not found near here' end
	if source and source.species then
		return 'Needs ID — not sure enough (' .. source.species .. '?)'
	end
	return 'Needs ID'
end

--- Group the plans by what they claim, so the preview reads as a summary.
local function tally(planned)
	local counts, order = {}, {}
	for _, entry in ipairs(planned) do
		local label = entry.claim
		if counts[label] == nil then
			counts[label] = 0
			order[#order + 1] = label
		end
		counts[label] = counts[label] + 1
	end
	table.sort(order, function(a, b)
		if counts[a] ~= counts[b] then return counts[a] > counts[b] end
		return a < b
	end)
	return counts, order
end

applyPlans = function(catalog, entries)
	local applied = 0
	catalog:withWriteAccessDo('Melampus: apply identifications', function()
		for _, entry in ipairs(entries) do
			local photo, plan = entry.photo, entry.plan
			if plan.rating ~= nil then photo:setRawMetadata('rating', plan.rating) end
			if plan.label ~= nil then photo:setRawMetadata('colorNameForLabel', plan.label) end
			if plan.pickStatus ~= nil then photo:setRawMetadata('pickStatus', plan.pickStatus) end
			for _, path in ipairs(plan.keywords) do
				local keyword = keywordFromPath(catalog, path)
				if keyword then photo:addKeyword(keyword) end
			end
			for field, value in pairs(plan.metadata or {}) do
				photo:setPropertyForPlugin(_PLUGIN, field, tostring(value))
			end
			applied = applied + 1
		end
	end, { timeout = 60 })
	return applied
end

LrTasks.startAsyncTask(function()
	LrFunctionContext.callWithContext('melampusImport', function(context)
		local prefs = LrPrefs.prefsForPlugin()
		local catalog = LrApplication.activeCatalog()

		local settings = Rules.defaultSettings()
		for key in pairs(settings) do
			if prefs[key] ~= nil then settings[key] = prefs[key] end
		end

		Log.info('=== import started ===')
		Log.info('results path: ' .. tostring(prefs.resultsPath))
		Log.info('dryRun=' .. tostring(settings.dryRun)
			.. ' writeKeywords=' .. tostring(settings.writeKeywords)
			.. ' writeMetadata=' .. tostring(settings.writeMetadata)
			.. ' force=' .. tostring(settings.force))

		local records, err, resolvedPath = readResults(prefs.resultsPath)
		if resolvedPath and resolvedPath ~= prefs.resultsPath then
			-- Remember what we found so Settings shows it and the next run is direct.
			prefs.resultsPath = resolvedPath
		end
		if not records then
			Log.error('could not read results: ' .. tostring(err))
			if err == 'FIRST_RUN' then
				-- §5.4.6: a first run should explain itself, not error.
				LrDialogs.message('Welcome to Melampus',
					'Melampus works out what species are in your photos, on your own Mac, '
					.. 'and then adds them to your photos as keywords.\n\n'
					.. 'It has not been told where those identifications are yet.\n\n'
					.. 'What to do:\n'
					.. '1.  Go to Library > Plug-in Extras > Melampus: Settings…\n'
					.. '2.  Under Step 1, click Choose… and pick the file\n'
					.. '     plugin_results.json in your melampus folder.\n'
					.. '3.  Leave "Preview only" ticked.\n'
					.. '4.  Come back here and run Identify Selected Photos again.\n\n'
					.. 'Nothing will be changed on your photos until you untick '
					.. '"Preview only" yourself.', 'info')
			else
				LrDialogs.message('Melampus', err, 'critical')
			end
			return
		end
		Log.info('records loaded: ' .. tostring(#records))

		local byFile = {}
		for _, record in ipairs(records) do
			if type(record) == 'table' and record.file then
				byFile[record.file] = record
				-- Results come from exported JPEGs; the catalog holds raws.
				-- Match on basename so 0A1A2475.jpg finds 0A1A2475.CR3.
				local stem = string.match(record.file, '^(.+)%.[^.]+$')
				if stem then byFile[stem] = record end
			end
		end

		local photos = catalog:getTargetPhotos()
		Log.info('photos selected: ' .. tostring(#photos))
		if #photos == 0 then
			Log.warn('nothing selected')
			LrDialogs.message('Melampus', 'Select some photos first.', 'info')
			return
		end

		local progress = LrProgressScope({
			title = settings.dryRun and 'Melampus: previewing changes'
				or 'Melampus: writing metadata',
			functionContext = context,
		})
		progress:setCancelable(true)

		-- ── phase 1: read and decide. No writes, yielding is fine here. ──────
		local planned, matched, unmatched = {}, 0, 0
		for index, photo in ipairs(photos) do
			if progress:isCanceled() then break end
			progress:setPortionComplete(index - 1, #photos * 2)

			local name = photo:getFormattedMetadata('fileName') or ''
			local stem = string.match(name, '^(.+)%.[^.]+$') or name
			local record = byFile[name] or byFile[stem]

			if record then
				matched = matched + 1
				local plan = Rules.planFor(normalise(record), photoState(photo), settings)
				if not Rules.isEmpty(plan) then
					local normalised = normalise(record)
					planned[#planned + 1] = {
						photo = photo, plan = plan, name = name,
						claim = describe(plan, normalised),
					}
				end
			else
				unmatched = unmatched + 1
				if unmatched <= 5 then
					Log.warn('no result for: ' .. name .. '  (looked for "' .. stem .. '")')
				end
			end
		end
		Log.info(string.format('matched %d, unmatched %d, with changes %d',
			matched, unmatched, #planned))

		-- Photos with no identification are the whole point of the tool, not an
		-- error to report. Offer to analyse them rather than shrugging.
		if unmatched > 0 and not progress:isCanceled() then
			local minutes = math.max(1, math.floor(unmatched * 7 / 60))
			local ask = LrDialogs.confirm('Melampus',
				string.format('%d of the %d selected photos have never been analysed.\n\n'
					.. 'Melampus can analyse them now on this Mac. Nothing is uploaded '
					.. 'anywhere.\n\nRoughly %d minute%s at about 7 seconds a photo. '
					.. 'You can cancel part way and keep whatever finished.',
					unmatched, #photos, minutes, minutes == 1 and '' or 's'),
				'Analyse them', 'Skip')

			if ask == 'ok' then
				local repo = Analyze.repoRoot()
				local toAnalyse = {}
				for _, photo in ipairs(photos) do
					local name = photo:getFormattedMetadata('fileName') or ''
					local stem = string.match(name, '^(.+)%.[^.]+$') or name
					if not (byFile[name] or byFile[stem]) then
						toAnalyse[#toAnalyse + 1] = photo
					end
				end

				-- §5.4.7: work in batches and write after each one, so stars and
				-- keywords appear as the run proceeds. A single pass over 700
				-- photos would show nothing for an hour and look frozen, which is
				-- indistinguishable from being frozen.
				local BATCH = Rules.batchSize(settings)
				local doneCount, appliedTotal = 0, 0

				for first = 1, #toAnalyse, BATCH do
					if progress:isCanceled() then break end
					local last = math.min(first + BATCH - 1, #toAnalyse)
					local batch = {}
					for i = first, last do batch[#batch + 1] = toAnalyse[i] end

					progress:setPortionComplete(doneCount, #toAnalyse)
					local workFolder = LrPathUtils.child(
						LrPathUtils.getStandardFilePath('temp'),
						'melampus-previews-' .. tostring(first))

					local exported = Analyze.exportPreviews(batch, workFolder, 1280, nil)
					if exported == 0 then
						Log.warn('no previews exported for batch starting at ' .. first)
					else
						local batchResults = LrPathUtils.child(workFolder, 'results.json')
						local ok, message = Analyze.run(repo, workFolder, batchResults)
						if not ok then
							LrDialogs.message('Melampus', message
								.. '\n\nPreviews kept at:\n' .. workFolder, 'critical')
							return
						end

						local fresh = readResults(batchResults)
						if fresh then
							-- Apply this batch immediately so the grid updates.
							local batchPlans = {}
							for _, record in ipairs(fresh) do
								if type(record) == 'table' and record.file then
									local stem = string.match(record.file, '^(.+)%.[^.]+$')
									for _, photo in ipairs(batch) do
										local pname = photo:getFormattedMetadata('fileName') or ''
										local pstem = string.match(pname, '^(.+)%.[^.]+$') or pname
										if pstem == stem then
											local plan = Rules.planFor(normalise(record),
												photoState(photo), settings)
											if not Rules.isEmpty(plan) then
												batchPlans[#batchPlans + 1] = { photo = photo, plan = plan }
											end
										end
									end
								end
							end

							if #batchPlans > 0 and Rules.shouldApply(settings) then
								applyPlans(catalog, batchPlans)
								appliedTotal = appliedTotal + #batchPlans
							elseif #batchPlans > 0 then
								for _, entry in ipairs(batchPlans) do
									planned[#planned + 1] = { photo = entry.photo, plan = entry.plan,
										name = entry.photo:getFormattedMetadata('fileName'),
										claim = 'newly analysed' }
								end
							end
						end

						if settings.keepPreviews ~= true then
							Analyze.cleanUp(workFolder)
						end
					end

					doneCount = doneCount + #batch
					Log.info(string.format('analysed %d of %d, applied %d so far',
						doneCount, #toAnalyse, appliedTotal))
				end

				if Rules.shouldApply(settings) then
					progress:done()
					LrDialogs.message('Melampus — done',
						string.format('Analysed %d photos and updated %d of them.%s\n\n'
							.. 'Stars and keywords appeared as each batch finished.',
							doneCount, appliedTotal,
							progress:isCanceled() and '\n\nCancelled — finished work was kept.' or ''),
						'info')
					return
				end
			end
		end

		-- ── preview, and offer to apply straight from it ────────────────────
		-- A preview you cannot act on is a dead end: you would have to go to
		-- Settings, untick a box, and run again. So the preview itself asks.
		if not Rules.shouldApply(settings) then
			local counts, order = tally(planned)
			local lines = {
				string.format('%d of your %d selected photos have identifications.',
					matched, #photos),
			}
			if unmatched > 0 then
				lines[#lines + 1] = string.format(
					'%d have none and will be left alone.', unmatched)
			end
			lines[#lines + 1] = ''

			if #planned == 0 then
				if matched > 0 then
					lines[#lines + 1] = 'Nothing to change — these photos already have '
						.. 'their Melampus keywords.'
				else
					lines[#lines + 1] = 'Nothing to change.'
				end
				progress:done()
				Log.info('preview: nothing to change')
				LrDialogs.message('Melampus', table.concat(lines, '\n'), 'info')
				return
			end

			lines[#lines + 1] = string.format('Melampus would tag %d photos:', #planned)
			lines[#lines + 1] = ''
			for _, label in ipairs(order) do
				lines[#lines + 1] = string.format('   %3d x  %s', counts[label], label)
			end
			lines[#lines + 1] = ''
			lines[#lines + 1] = 'Species keywords are only added when it is confident '
				.. 'and the species occurs near where you shot. Everything else '
				.. 'becomes "Needs ID" for you to look at.'

			progress:done()
			Log.info('preview: ' .. #planned .. ' photos would change')

			local choice = LrDialogs.confirm('Melampus — preview',
				table.concat(lines, '\n'), 'Add these keywords', 'Not yet')
			if choice ~= 'ok' then
				Log.info('user declined to apply')
				return
			end
			Log.info('user approved from preview; applying')

			-- Re-open a scope for the writing phase, since the preview closed it.
			progress = LrProgressScope({
				title = 'Melampus: adding keywords',
				functionContext = context,
			})
			progress:setCancelable(true)
		end

		-- ── phase 2: apply. Chunked, and nothing async inside the gate. ─────
		local written, chunkStart = 0, 1
		local keywordsApplied, keywordsFailed, fieldsApplied = 0, 0, 0
		while chunkStart <= #planned do
			if progress:isCanceled() then break end
			local chunkStop = math.min(chunkStart + CHUNK - 1, #planned)

			catalog:withWriteAccessDo('Melampus: apply identifications', function()
				for i = chunkStart, chunkStop do
					local entry = planned[i]
					local photo, plan = entry.photo, entry.plan

					if plan.rating ~= nil then photo:setRawMetadata('rating', plan.rating) end
					if plan.label ~= nil then photo:setRawMetadata('colorNameForLabel', plan.label) end
					if plan.pickStatus ~= nil then photo:setRawMetadata('pickStatus', plan.pickStatus) end

					local appliedHere = 0
					for _, path in ipairs(plan.keywords) do
						local keyword = keywordFromPath(catalog, path)
						if keyword then
							photo:addKeyword(keyword)
							appliedHere = appliedHere + 1
							keywordsApplied = keywordsApplied + 1
						else
							keywordsFailed = keywordsFailed + 1
						end
					end

					for field, value in pairs(plan.metadata or {}) do
						photo:setPropertyForPlugin(_PLUGIN, field, tostring(value))
						fieldsApplied = fieldsApplied + 1
					end

					-- Only count a photo as changed if something actually changed on
					-- it. The previous counter incremented per photo processed, so a
					-- run that silently wrote nothing still reported full success.
					if appliedHere > 0 or next(plan.metadata or {}) ~= nil then
						written = written + 1
					end
				end
			end, { timeout = 60 })

			progress:setPortionComplete(#photos + chunkStop, #photos * 2)
			chunkStart = chunkStop + 1
		end

		-- Timestamp separately: it is plugin-private, so it stays off the undo
		-- stack rather than cluttering it with a bookkeeping entry.
		catalog:withPrivateWriteAccessDo(function()
			local stamp = os.date('%Y-%m-%dT%H:%M:%S')
			for i = 1, math.min(written, #planned) do
				planned[i].photo:setPropertyForPlugin(_PLUGIN, 'processedAt', stamp)
			end
		end)

		Log.info(string.format('photos changed: %d | keywords applied: %d, failed: %d | fields: %d',
			written, keywordsApplied, keywordsFailed, fieldsApplied))
		progress:done()
		local suffix = progress:isCanceled()
			and '\n\nCancelled — completed work was kept.' or ''
		local body = string.format(
			'Changed %d photos.\n\n%d keywords added.\n%d panel fields written.\n',
			written, keywordsApplied, fieldsApplied)
		if keywordsFailed > 0 then
			body = body .. string.format(
				'\nWARNING: %d keywords could not be created. The keyword list may be '
				.. 'incomplete. See the log:\n%s\n', keywordsFailed, Log.path())
		end
		if unmatched > 0 then
			body = body .. string.format(
				'\n%d selected photos had no identification and were left untouched.\n',
				unmatched)
		end
		body = body .. '\nCheck the Keyword List panel for "Melampus".' .. suffix
		LrDialogs.message('Melampus — done', body, 'info')
	end)
end)
