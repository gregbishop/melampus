--[[ Export the corrections you made during review (§6.4).

     Two payoffs, both stated in the spec: prompt tuning now, and eventually a
     fine-tune on local taxa. The file it writes is the same shape the Python
     side's ingest_corrections.py already consumes, so the loop closes without a
     translation step.

     Reads only. It never modifies the catalog. ]]
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrPathUtils = import 'LrPathUtils'
local LrTasks = import 'LrTasks'

local function escape(text)
	text = tostring(text or '')
	text = string.gsub(text, '\\', '\\\\')
	text = string.gsub(text, '"', '\\"')
	text = string.gsub(text, '\n', '\\n')
	text = string.gsub(text, '\t', '\\t')
	return text
end

LrTasks.startAsyncTask(function()
	local catalog = LrApplication.activeCatalog()
	local photos = catalog:getAllPhotos()

	local rows, counts = {}, { confirmed = 0, wrong = 0, uncertain = 0, unreviewed = 0 }

	for _, photo in ipairs(photos) do
		local verdict = photo:getPropertyForPlugin(_PLUGIN, 'verdict')
		if verdict and verdict ~= '' then
			counts[verdict] = (counts[verdict] or 0) + 1
			-- Only reviewed photos are worth exporting; unreviewed is not a signal.
			if verdict ~= 'unreviewed' then
				local proposed = photo:getPropertyForPlugin(_PLUGIN, 'species')
				local corrected = photo:getPropertyForPlugin(_PLUGIN, 'correction')
				rows[#rows + 1] = string.format(
					'{"file":"%s","encounter":%s,"verdict":"%s","model_call":"%s","corrected_to":%s}',
					escape(photo:getFormattedMetadata('fileName')),
					tostring(tonumber(photo:getPropertyForPlugin(_PLUGIN, 'encounter')) or 'null'),
					escape(verdict),
					escape(proposed),
					(corrected and corrected ~= '')
						and ('"' .. escape(corrected) .. '"') or 'null')
			end
		end
	end

	if #rows == 0 then
		LrDialogs.message('Melampus: Log Corrections',
			'No reviewed photos found yet.\n\nSet a verdict in the Melampus metadata '
			.. 'panel while reviewing, then run this again.', 'info')
		return
	end

	local target = LrDialogs.runSavePanel {
		title = 'Save Melampus corrections',
		requiredFileType = 'json',
		canCreateDirectories = true,
	}
	if not target then return end

	local handle, err = io.open(target, 'w')
	if not handle then
		LrDialogs.message('Melampus', 'Could not write:\n' .. tostring(err), 'critical')
		return
	end
	handle:write('{"corrections":[\n', table.concat(rows, ',\n'), '\n]}\n')
	handle:close()

	LrDialogs.message('Melampus: Log Corrections',
		string.format('Wrote %d reviewed photos.\n\nconfirmed %d, corrected %d, uncertain %d\n\n'
			.. 'Feed it back with:\n  tools/ingest_corrections.py',
			#rows, counts.confirmed or 0, counts.wrong or 0, counts.uncertain or 0), 'info')
end)
