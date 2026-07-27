--[[
Custom metadata schema.

CLAUDE.md §5.2 asks for the numeric detail to live in custom fields rather than
keywords — confidence, agreement, range flag, model and version — so the keyword
tree stays browsable and the numbers stay queryable.

Every field is read-only except `verdict` and `correction`, which are the two you
actually touch during review. Read-only here means Lightroom will not let a stray
keystroke overwrite a computed value; the plugin still writes them.

`model` and `schemaVersion` are recorded per photo on purpose: when a better
model lands you need to know what was tagged by what in order to decide what
merits reprocessing.
--]]

return {
	metadataFieldsForPhotos = {
		-- ── what the model proposed ─────────────────────────────────────────
		{
			id = 'species',
			title = 'Melampus Species',
			dataType = 'string',
			searchable = true,
			browsable = true,
			readOnly = true,
		},
		{
			id = 'scientificName',
			title = 'Scientific Name',
			dataType = 'string',
			searchable = true,
			browsable = true,
			readOnly = true,
		},
		{
			id = 'alternates',
			title = 'Or possibly',
			dataType = 'string',
			readOnly = true,
		},
		{
			id = 'taxon',
			title = 'Taxon',
			dataType = 'string',
			searchable = true,
			browsable = true,
			readOnly = true,
		},

		-- ── how much to trust it ────────────────────────────────────────────
		-- Stored as strings because the SDK has no numeric field type. Values are
		-- zero-padded so smart-collection string comparisons sort correctly.
		{
			id = 'confidence',
			title = 'Confidence',
			dataType = 'string',
			searchable = true,
			browsable = true,
			readOnly = true,
		},
		{
			id = 'burstAgreement',
			title = 'Burst Agreement',
			dataType = 'string',
			searchable = true,
			browsable = true,
			readOnly = true,
		},
		{
			id = 'rangeFlag',
			title = 'Range Flag',
			dataType = 'string',
			searchable = true,
			browsable = true,
			readOnly = true,
		},
		{
			id = 'encounter',
			title = 'Encounter',
			dataType = 'string',
			searchable = true,
			browsable = true,
			readOnly = true,
		},

		-- ── your review ─────────────────────────────────────────────────────
		-- The only editable fields. Enum values are fixed at definition time,
		-- which is fine here because a verdict really is a closed set.
		{
			id = 'verdict',
			title = 'Review Verdict',
			dataType = 'enum',
			searchable = true,
			browsable = true,
			values = {
				{ value = 'unreviewed', title = 'Unreviewed' },
				{ value = 'confirmed',  title = 'Confirmed correct' },
				{ value = 'wrong',      title = 'Wrong — see correction' },
				{ value = 'uncertain',  title = "Can't tell" },
			},
			allowPluginToSetOnImport = true,
		},
		{
			id = 'correction',
			title = 'Correct Species',
			dataType = 'string',
			searchable = true,
			browsable = true,
		},

		-- ── provenance ──────────────────────────────────────────────────────
		{
			id = 'model',
			title = 'Model',
			dataType = 'string',
			searchable = true,
			browsable = true,
			readOnly = true,
		},
		{
			id = 'processedAt',
			title = 'Processed',
			dataType = 'string',
			readOnly = true,
		},
		{
			id = 'schemaVersion',
			title = 'Schema Version',
			dataType = 'string',
			readOnly = true,
		},
	},

	schemaVersion = 1,

	-- Called when the stored schema predates this one. Returning without doing
	-- anything is safe for version 1; future versions migrate here rather than
	-- silently reinterpreting old values.
	updateFromEarlierSchemaVersion = function(catalog, previousSchemaVersion, progressScope)
		return
	end,
}
