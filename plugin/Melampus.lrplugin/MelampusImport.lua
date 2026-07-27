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

local Json = require 'MelampusJson'
local Rules = require 'MelampusRules'

local PLUGIN_ID = 'net.gregbishop.melampus'
local CHUNK = 100 -- photos per write transaction, so a crash loses little

local function readResults(path)
	if not path or path == '' then
		return nil, 'No results file configured. Set one in Melampus: Settings.'
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
		abstain = abstain,
		-- These three are supplied by the encounter-aware exporter. Absent is
		-- handled: Rules treats a missing agreement as "unknown", not "failed".
		burstAgreement = record.burst_agreement,
		rangeFlag = record.range_flag == true,
		encounter = record.encounter,
		quality = record.quality,
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
			-- returnExisting = true, so re-running never duplicates the tree.
			parent = catalog:createKeyword(name, {}, false, parent, true)
		end
	end
	return parent
end

local function describe(plan)
	local bits = {}
	if plan.rating then bits[#bits + 1] = 'rating ' .. plan.rating end
	if plan.label then bits[#bits + 1] = 'label ' .. plan.label end
	if plan.pickStatus then bits[#bits + 1] = 'flag ' .. plan.pickStatus end
	if #plan.keywords > 0 then bits[#bits + 1] = #plan.keywords .. ' keywords' end
	local fields = 0
	for _ in pairs(plan.metadata or {}) do fields = fields + 1 end
	if fields > 0 then bits[#bits + 1] = fields .. ' fields' end
	return table.concat(bits, ', ')
end

LrTasks.startAsyncTask(function()
	LrFunctionContext.callWithContext('melampusImport', function(context)
		local prefs = LrPrefs.prefsForPlugin()
		local catalog = LrApplication.activeCatalog()

		local settings = Rules.defaultSettings()
		for key in pairs(settings) do
			if prefs[key] ~= nil then settings[key] = prefs[key] end
		end

		local records, err = readResults(prefs.resultsPath)
		if not records then
			LrDialogs.message('Melampus', err, 'critical')
			return
		end

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
		if #photos == 0 then
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
					planned[#planned + 1] = { photo = photo, plan = plan, name = name }
				end
			else
				unmatched = unmatched + 1
			end
		end

		-- ── dry run stops here ──────────────────────────────────────────────
		if not Rules.shouldApply(settings) then
			local lines = {
				string.format('DRY RUN — nothing was written.\n'),
				string.format('%d selected, %d matched, %d without results.', #photos, matched, unmatched),
				string.format('%d photos would change.\n', #planned),
			}
			for i = 1, math.min(#planned, 15) do
				lines[#lines + 1] = '  ' .. planned[i].name .. ': ' .. describe(planned[i].plan)
			end
			if #planned > 15 then
				lines[#lines + 1] = string.format('  … and %d more.', #planned - 15)
			end
			lines[#lines + 1] = '\nTurn off Dry Run in Melampus: Settings to apply.'
			progress:done()
			LrDialogs.message('Melampus — dry run', table.concat(lines, '\n'), 'info')
			return
		end

		-- ── phase 2: apply. Chunked, and nothing async inside the gate. ─────
		local written, chunkStart = 0, 1
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

					for _, path in ipairs(plan.keywords) do
						local keyword = keywordFromPath(catalog, path)
						if keyword then photo:addKeyword(keyword) end
					end

					for field, value in pairs(plan.metadata or {}) do
						photo:setPropertyForPlugin(_PLUGIN, field, tostring(value))
					end
					written = written + 1
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

		progress:done()
		local suffix = progress:isCanceled()
			and '\n\nCancelled — completed work was kept.' or ''
		LrDialogs.message('Melampus',
			string.format('%d photos updated of %d selected.\n%d had no results.%s',
				written, #photos, unmatched, suffix), 'info')
	end)
end)
