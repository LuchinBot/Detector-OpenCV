import xml.etree.ElementTree as ET

# Crear el elemento raíz
root = ET.Element("opencv_storage")

# Agregar un elemento de clasificador en cascada
cascade = ET.SubElement(root, "cascade", type_id="opencv-cascade-classifier")

# Agregar elementos dentro del clasificador en cascada
stageType = ET.SubElement(cascade, "stageType")
stageType.text = "BOOST"

featureType = ET.SubElement(cascade, "featureType")
featureType.text = "HAAR"

# Agregar más elementos según sea necesario...

# Crear el árbol XML
tree = ET.ElementTree(root)

# Guardar el archivo XML
tree.write("clasificador_cascada.xml")
