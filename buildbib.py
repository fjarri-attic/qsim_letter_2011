import sys, re

source = sys.argv[1]
dest = sys.argv[2]

f = open(source)
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
	entry_key = mo.group(2)

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

	entries.append((entry_type, entry_key, entry_fields))

# write bib file
f = open(dest, 'w')
for entry_type, entry_key, entry_fields in entries:
	f.write('@' + entry_type + '{' + entry_key + ',\n')
	for field in entry_fields:
		f.write('\t' + field + ' = {' + entry_fields[field] + '},\n')
	f.write('}\n\n')

f.close()
