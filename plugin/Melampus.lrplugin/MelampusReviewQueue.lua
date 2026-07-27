--[[ Create the smart collections and jump to the review queue (§5.4.3).

     Review happens in Lightroom's own grid and loupe; this only builds the
     collections that route photos there. Creating a collection that already
     exists is a no-op, so running this repeatedly is safe. ]]
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrTasks = import 'LrTasks'

local PLUGIN_ID = 'net.gregbishop.melampus'

local function criteria(field, operation, value)
	return {
		{
			criteria = 'sdktext:' .. PLUGIN_ID .. '.' .. field,
			operation = operation,
			value = value,
		},
		combine = 'intersect',
	}
end

LrTasks.startAsyncTask(function()
	local catalog = LrApplication.activeCatalog()
	local made, existing = {}, {}

	catalog:withWriteAccessDo('Melampus: create smart collections', function()
		local set = catalog:createCollectionSet('Melampus', nil, true)

		local wanted = {
			{ name = 'Needs Review',          spec = criteria('verdict', '==', 'unreviewed') },
			{ name = 'Notable — Out of Range', spec = criteria('rangeFlag', '==', 'out-of-range') },
			{ name = 'Confirmed',             spec = criteria('verdict', '==', 'confirmed') },
			{ name = 'Corrected',             spec = criteria('verdict', '==', 'wrong') },
		}

		for _, entry in ipairs(wanted) do
			-- returnExisting = true, so this is idempotent. It returns the
			-- collection either way, which is why we do not try to report
			-- "created" versus "already there" -- we cannot tell.
			local collection = catalog:createSmartCollection(entry.name, entry.spec, set, true)
			if collection then made[#made + 1] = entry.name end
		end
	end, { timeout = 30 })

	local message = string.format('%d smart collections ready under the Melampus set:\n  %s\n\n',
			#made, table.concat(made, ', '))
		.. 'Open "Needs Review", turn on the Melampus panel in the Metadata pane,\n'
		.. 'and work through them in loupe view. Set the verdict as you go and the\n'
		.. 'collection empties itself.'
	LrDialogs.message('Melampus: Review Queue', message, 'info')
end)
