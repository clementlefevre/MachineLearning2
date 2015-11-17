import io
import sys


if len(sys.argv) < 2:
    print("You must enter a .vcf file name")
else:
    with io.open(sys.argv[1], encoding="UTF-8") as file:

        contacts = []
        contact = []
        line = file.readline()
        while(line):
            #line = line.encode(sys.stdout.encoding, errors='replace')
            if line == 'BEGIN:VCARD\n':
                contact = []
            if len(line.split(':')) == 1:
                # the value has continued onto the next line, add it to the previous one
                contact[-1]['value'] = '{0}{1}'.format(contact[-1]['value'], line)
            else:
                try:
                    key = line.split(':')[0]
                    value  = ':'.join(line.split(':')[1:])
                    # replace google values with skype friendly ones
                    if 'TYPE=CELL' in key:
                        print('TEST')
                        key = 'X-SKYPE-PSTNNUMBER'
                        value = '+{0}'.format(value.strip(' '))
                        value = value.replace('-', '')
                        value = value.replace(' ', '')
                        value = value.replace('(', '')
                        value = value.replace(')', '')
                    elif 'FN' == key:
                        print('TEST1')
                        key = 'X-SKYPE-DISPLAYNAME'
                    print(contact)
                    contact.append({
                        'key': key,
                        'value': value
                    })
                except Exception:
                    pass
            if line in ['END:VCARD', 'END:VCARD\n']:
                contacts.append(contact)
            line = file.readline()

        
    with io.open('output.vcf', 'w', encoding="utf8") as file:
        for contact in contacts:
            for arg in contact:
                try:
                    file.write('{0}:{1}'.format(arg['key'], arg['value']))
                except Exception :
                    pass