--[[ Metadata panel layout.

     Order matters: the proposed species and its alternates sit directly above
     the two editable fields, so the whole review decision is one glance and one
     dropdown without scrolling. Provenance sits at the bottom where it is
     available but out of the way. ]]
return {
	id = 'melampusTagset',
	title = 'Melampus',
	items = {
		'com.adobe.label',
		'com.adobe.title',
		'com.adobe.separator',

		'net.gregbishop.melampus.species',
		'net.gregbishop.melampus.scientificName',
		'net.gregbishop.melampus.alternates',
		'net.gregbishop.melampus.taxon',
		'com.adobe.separator',

		'net.gregbishop.melampus.verdict',
		'net.gregbishop.melampus.correction',
		'com.adobe.separator',

		'net.gregbishop.melampus.confidence',
		'net.gregbishop.melampus.burstAgreement',
		'net.gregbishop.melampus.rangeFlag',
		'net.gregbishop.melampus.encounter',
		'com.adobe.separator',

		'net.gregbishop.melampus.model',
		'net.gregbishop.melampus.processedAt',
	},
}
