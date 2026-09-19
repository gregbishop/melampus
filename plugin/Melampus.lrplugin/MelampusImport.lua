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

--- The records in a results file, or an empty list when none is configured.
-- No results file is the normal state of a fresh install, not an error: it
-- means nothing has been analysed yet, and the selection goes to the analyse
-- offer below, which runs the executable beside the plugin. A results file is
-- only for identifications produced elsewhere (melampus-id --plugin-out).
local function readResults(path)
	if not path or path == '' then
		return {}
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
	return data
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
		behaviour = type(ident) == 'table' and ident.behavior or nil,
		ageSex = type(ident) == 'table' and ident.age_sex or nil,
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

--- Read the catalog back and confirm a plan actually landed.
-- The worst failure this project has had was a run reporting 1301 photos updated
-- while the catalog held keywords for about 37. A counter incremented next to a
-- write is not evidence the write happened; reading it back is.
local function verifyApplied(entries)
	local verified, failed = 0, {}
	for _, entry in ipairs(entries) do
		local photo, plan = entry.photo, entry.plan
		local ok = true

		if plan.rating ~= nil and photo:getRawMetadata('rating') ~= plan.rating then
			ok = false
		end
		if ok and #plan.keywords > 0 then
			local present = {}
			for _, kw in ipairs(photo:getRawMetadata('keywords') or {}) do
				present[kw:getName()] = true
			end
			for _, path in ipairs(plan.keywords) do
				local leaf = string.match(path, '([^>]+)$')
				leaf = string.gsub(leaf or '', '^%s*(.-)%s*$', '%1')
				if leaf ~= '' and not present[leaf] then ok = false end
			end
		end

		if ok then
			verified = verified + 1
		else
			failed[#failed + 1] = entry
		end
	end
	return verified, failed
end

--- Total keywords across a set of plans, for honest reporting.
local function keywordCount(entries)
	local n = 0
	for _, entry in ipairs(entries) do n = n + #entry.plan.keywords end
	return n
end

--- Write one batch of plans, read the catalog back, and retry what did not land.
--
-- This is the only place in the plugin that writes identifications. There used to
-- be a second, near-identical loop for the apply-from-results path, which meant
-- the verification below protected only half the runs — and the half it did not
-- protect was the one most people use.
--
-- Returns a stats table describing what the catalog *confirms*, not what was
-- attempted. Callers must report `verified`; reporting the plan count is how a run
-- once claimed 1301 updates against a catalog holding keywords for about 37.
applyPlans = function(catalog, entries)
	local stats = {
		verified = 0, failed = 0,
		keywords = 0, keywordsFailed = 0, fields = 0,
		applied = {},
	}
	if #entries == 0 then return stats end

	local function write(batch, full)
		catalog:withWriteAccessDo('Melampus: apply identifications', function()
			for _, entry in ipairs(batch) do
				local photo, plan = entry.photo, entry.plan
				if plan.rating ~= nil then photo:setRawMetadata('rating', plan.rating) end
				if full then
					if plan.label ~= nil then
						photo:setRawMetadata('colorNameForLabel', plan.label)
					end
					if plan.pickStatus ~= nil then
						photo:setRawMetadata('pickStatus', plan.pickStatus)
					end
				end
				for _, path in ipairs(plan.keywords) do
					local keyword = keywordFromPath(catalog, path)
					if keyword then
						photo:addKeyword(keyword)
					elseif full then
						-- Count creation failures once, on the first pass only.
						stats.keywordsFailed = stats.keywordsFailed + 1
					end
				end
				if full then
					for field, value in pairs(plan.metadata or {}) do
						photo:setPropertyForPlugin(_PLUGIN, field, tostring(value))
						stats.fields = stats.fields + 1
					end
				end
			end
		end, { timeout = 60 })
	end

	write(entries, true)

	-- Self-heal: read back, and retry anything that did not land. One repeat is
	-- enough for transient contention; a second failure is a real problem and is
	-- reported rather than swallowed.
	local _, failed = verifyApplied(entries)
	if #failed > 0 then
		Log.warn(string.format('%d of %d writes did not land; retrying', #failed, #entries))
		write(failed, false)
		local _, stillFailed = verifyApplied(failed)
		failed = stillFailed
		if #failed > 0 then
			Log.error(string.format('%d writes failed twice and were not applied', #failed))
			for i = 1, math.min(#failed, 5) do
				Log.error('  unwritten: '
					.. tostring(failed[i].photo:getFormattedMetadata('fileName')))
			end
		end
	end

	local unwritten = {}
	for _, entry in ipairs(failed) do unwritten[entry] = true end
	for _, entry in ipairs(entries) do
		if not unwritten[entry] then stats.applied[#stats.applied + 1] = entry end
	end
	stats.verified = #stats.applied
	stats.failed = #failed
	stats.keywords = keywordCount(entries) - keywordCount(failed)

	Log.info(string.format('attempted %d, verified in catalog %d', #entries, stats.verified))
	return stats
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

		local records, err = readResults(prefs.resultsPath)
		if not records then
			Log.error('could not read results: ' .. tostring(err))
			LrDialogs.message('Melampus', err, 'critical')
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
					.. 'Melampus can analyse them now on this computer. Nothing is uploaded '
					.. 'anywhere.\n\nRoughly %d minute%s at about 7 seconds a photo. '
					.. 'You can cancel part way and keep whatever finished.',
					unmatched, #photos, minutes, minutes == 1 and '' or 's'),
				'Analyse them', 'Skip')

			if ask == 'ok' then
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
						local ok, message = Analyze.run(workFolder, batchResults,
							settings.profile, settings.engine)
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
								-- Count what the catalog confirms, not what we intended.
								appliedTotal = appliedTotal + applyPlans(catalog, batchPlans).verified
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
		local confirmed = {}
		while chunkStart <= #planned do
			if progress:isCanceled() then break end
			local chunkStop = math.min(chunkStart + CHUNK - 1, #planned)

			local chunk = {}
			for i = chunkStart, chunkStop do chunk[#chunk + 1] = planned[i] end

			-- One shared implementation with the analyse path: write, read back,
			-- retry once, then report only what the catalog confirms.
			local stats = applyPlans(catalog, chunk)
			written = written + stats.verified
			keywordsApplied = keywordsApplied + stats.keywords
			keywordsFailed = keywordsFailed + stats.keywordsFailed
			fieldsApplied = fieldsApplied + stats.fields
			for _, entry in ipairs(stats.applied) do confirmed[#confirmed + 1] = entry end

			progress:setPortionComplete(#photos + chunkStop, #photos * 2)
			chunkStart = chunkStop + 1
		end

		-- Timestamp separately: it is plugin-private, so it stays off the undo
		-- stack rather than cluttering it with a bookkeeping entry. Only photos
		-- whose writes were verified get stamped — marking a photo processed when
		-- nothing landed would make the failure permanent, because the next run
		-- would skip it as already done.
		catalog:withPrivateWriteAccessDo(function()
			local stamp = os.date('%Y-%m-%dT%H:%M:%S')
			for _, entry in ipairs(confirmed) do
				entry.photo:setPropertyForPlugin(_PLUGIN, 'processedAt', stamp)
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
