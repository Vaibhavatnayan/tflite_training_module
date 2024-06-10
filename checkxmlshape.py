import xml.etree.ElementTree as ET

# Path to the XML file
xml_file_path ='./dataset/train_labels/mobile-tower-set-up-at-16000-feet-in-ladakh-1-_webp.rf.7e7d46f5937900c749979e01dd919e79.xml'

# Parse the XML file
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Print the root element tag
print("Root Element Tag:", root.tag)

# Print attributes of the root element
print("Root Element Attributes:", root.attrib)

# Print child elements
print("Child Elements:")
for child in root:
    print(child.tag, child.attrib)

# Print text content of specific elements
for elem in root.findall('.//some/element'):
    print("Text Content of Element:", elem.text)

# Or any other operations you want to perform on the XML data
