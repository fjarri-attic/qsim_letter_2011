import sys, re
from unidecode import unidecode
import codecs

source = sys.argv[1]
dest = sys.argv[2]

f = codecs.open(source, 'r', 'utf-8')
text = f.read()
f.close()

raw_entries = ('\n' + text).split('\n@')[1:]

# parse bib file
entries = []
for raw_entry in raw_entries:
	lines = raw_entry.split('\n')

	# read header
	mo = re.match(r'^(\w+)\{(.*),$', lines[0])
	entry_type = mo.group(1)
	entry_key = unidecode(mo.group(2))

	# read fields
	entry_fields = {}
	for line in lines[1:]:

		if line == '}':
			break

		mo = re.match(r'^([\w-]+)\s+=\s+(.*?),?$', line)

		value = mo.group(2)
		if value[0] == '{':
			assert value[-1] == '}'
			value = value[1:-1]

		entry_fields[mo.group(1)] = value

	# remove unwanted fields
	for field in ('file', 'keywords', 'mendeley-tags', 'abstract'):
		if field in entry_fields:
			del entry_fields[field]

	# fix Arxiv references
	if 'arxivId' in entry_fields:
		if 'journal' in entry_fields:
		# already published, no need to cite arxiv
			del entry_fields['arxivId']
		else:
			aid = entry_fields['arxivId']
			del entry_fields['arxivId']

			entry_fields['archivePrefix'] = 'arXiv'
			entry_fields['eprint'] = aid[:4] + aid[5:]
			entry_fields['SLACcitation'] = "%%CITATION=" + aid[:4] + aid[5:] + ";%%"

	entries.append((entry_type, entry_key, entry_fields))

# prepare entries for PRL
for entry_type, entry_key, entry_fields in entries:

	if entry_type != 'article':
		continue

	# fix special symbols in authors
	entry_fields['author'] = entry_fields['author'].replace('\\o ', '{\\o}')

	# truncate author list
	authors = entry_fields['author'].split(' and ')
	if len(authors) >= 4:
		entry_fields['author'] = authors[0] + ' and others'

	# replace journal names with abbreviations
	abbreviations = {
		'Physical Review A': 'Phys. Rev. A',
		'Journal of Physics B: Atomic, Molecular and Optical Physics': 'J. Phys. B',
		'Physical Review Letters': 'Phys. Rev. Lett.',
		'The European Physical Journal B': 'Eur. Phys. J. B',
		'Europhysics Letters (EPL)': 'Europhys. Lett.',
	}

	if 'journal' in entry_fields and entry_fields['journal'] in abbreviations:
		entry_fields['journal'] = abbreviations[entry_fields['journal']]


# write bib file
f = open(dest, 'w')
for entry_type, entry_key, entry_fields in entries:
	f.write('@' + entry_type + '{' + entry_key + ',\n')
	for field in entry_fields:
		f.write('\t' + field + ' = {' + entry_fields[field] + '},\n')
	f.write('}\n\n')

f.close()
