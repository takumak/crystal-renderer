import sys, os
import math
import pylada.crystal.read
import numpy as np
from numpy.linalg import norm
import colorsys
from xml.etree import ElementTree as ET
svg_ns = 'http://www.w3.org/2000/svg'
ET.register_namespace('svg', svg_ns)
ET.register_namespace('inkscape', 'http://www.inkscape.org/namespaces/inkscape')


atom_radius = 40
atom_colors = {
  'O':  (0.62745098, 0.03529412, 0.03529412),
  'Sm': (0.03529412, 0.10196078, 0.62745098),
  'Fe': (0.62745098, 0.41568627, 0.03529412),
  'As': (0.24313725, 0.62745098, 0.03529412)
}
atom_border_width = 3
connections = {
  'Fe-As': 3,
  'O-Sm': 3
}
connection_width = 10
connection_style = 'fill:#ccc;stroke:#888;stroke-width:2'
camera_direction = 1, 0.3, 0.1 # hkl
render_size = 1000, 1000
# input filename
filename_poscar = 'SmFeAsO.vasp'
# output filename
filename_svg = 'SmFeAsO.svg'


normalize = lambda v: np.array(v)/norm(v)


def in_unitcell(r, cell):
  abc = [np.dot(r, e/norm(e))/norm(e) for e in cell]
  return np.min(abc) > -0.01 and np.max(abc) < 1.01


def search_atoms(structure):
  from pylada.crystal import neighbors

  atoms = []
  conns = []

  def atom_id(pos):
    for i, a, p in atoms:
      if np.all(norm(p - pos) < 1):
        return i
    return -1

  def add_atom(structure, atom, pos):
    if atom_id(pos) >= 0:
      return

    aid = len(atoms)
    atoms.append((aid, atom.type, pos))

    for atom_, pos_, dist_ in neighbors(structure, 20, pos):
      pos_ += pos

      bond = False
      cname = '%s-%s' % (atom.type, atom_.type)
      if cname in connections and dist_ <= connections[cname]:
        bond = True

      aid_ = atom_id(pos_)
      if aid_ < 0:
        if in_unitcell(pos_, structure.cell):
          aid_ = add_atom(structure, atom_, pos_)
        elif bond:
          aid_ = len(atoms)
          atoms.append((aid_, atom_.type, pos_))
        else:
          continue

      if bond:
        pair = set([aid, aid_])
        if pair not in conns:
          conns.append(pair)

    return aid

  add_atom(structure, structure[0], structure[0].pos)
  return atoms, conns


def create_defs(root):
  defs = ET.SubElement(root, 'defs')
  for name, col1 in atom_colors.items():
    h, s, v = colorsys.rgb_to_hsv(*col1)
    col0 = colorsys.hsv_to_rgb(h, 0.6, 1)

    cc0 = '#%02x%02x%02x' % tuple(map(int, np.array(col0)*255.))
    cc1 = '#%02x%02x%02x' % tuple(map(int, np.array(col1)*255.))
    grad = ET.SubElement(defs, 'radialGradient')
    grad.set('id', 'grad-%s' % name)
    grad.set('cx', '40%')
    grad.set('cy', '60%')
    grad.set('r', '50%')
    for off, cc in enumerate([cc0, cc1]):
      st = ET.SubElement(grad, 'stop')
      st.set('offset', str(off))
      st.set('style', 'stop-color:%s' % cc)


def create_atoms(root, atoms, conns, scale, translate):
  aid_map = {aid:(atom, pos) for aid, atom, pos in atoms}
  for aid, atom, pos in atoms:
    h, s, v = colorsys.rgb_to_hsv(*atom_colors[atom])
    sc = '#%02x%02x%02x' % tuple([int(a*255) for a in colorsys.hsv_to_rgb(h, s, 0.4)])

    cx, cy, cz = pos*scale + translate
    insidx = len(root)
    circ = ET.SubElement(root, 'circle')
    circ.set('x', '0')
    circ.set('y', '0')
    circ.set('r', '%g' % atom_radius)
    circ.set('style', 'fill:url(#grad-%s);stroke:%s;stroke-width:%g' % (atom, sc, atom_border_width))
    circ.set('transform', 'translate(%g,%g)' % (cx, cy))

    done = []
    for pair in conns:
      a1, a2 = pair
      if a2 == aid:
        t = a1
        a1 = a2
        a2 = t
      elif a1 != aid:
        continue

      p1 = pos
      p2 = aid_map[a2][1]

      p1 = p1 + normalize(p2-p1) * (atom_radius+atom_border_width/2)/scale
      p2 = p2 + normalize(p1-p2) * (atom_radius+atom_border_width/2)/scale

      n = p1 - p2
      t1 = np.arctan2(n[1], n[0])
      t0 = t1 - np.pi/2
      n = np.array([np.cos(t0), np.sin(t0)])

      r = connection_width/2/scale
      d = [p2[:2] + n*r, p1[:2] + n*r]
      s = np.dot(normalize(p1 - p2), (0, 0, 1))
      rot = np.array([[np.cos(t1), -np.sin(t1)],
                      [np.sin(t1), np.cos(t1)]])
      for theta in np.linspace(-np.pi/2, np.pi/2, 20):
        v = np.array([np.cos(theta)*s, np.sin(theta)])
        d.append(p1[:2]+r*np.dot(rot, v))
      d += [p1[:2] - n*r, p2[:2] - n*r]

      path = ET.SubElement(root, 'path')
      path.set('d', 'M ' + ' '.join(['%g,%g' % tuple(p*scale+translate[:2]) for p in d]) + ' z')
      path.set('style', connection_style)

      done.append(pair)

    for pair in done:
      conns.remove(pair)


def render():
  structure = pylada.crystal.read.poscar(filename_poscar)
  atoms, conns = search_atoms(structure)
  center = np.sum([pos for aid, atom, pos in atoms], axis=0)/len(atoms)

  camp = center + camera_direction*center*2
  camd = normalize(camera_direction)
  camx = -normalize(np.cross(camd, (0, 0, 1)))
  camy = np.cross(camd, camx)
  camz = -camd

  def cam_geom(pos):
    r = pos - camp
    return np.array([np.dot(r, e) for e in (camx, camy, camz)])

  atoms = [(aid, atom, cam_geom(pos)) for aid, atom, pos in atoms]
  atoms = list(sorted(atoms, key=lambda p: p[2][2], reverse=True))

  atomx = [pos[0] for aid, atom, pos in atoms]
  atomy = [pos[1] for aid, atom, pos in atoms]
  minx, maxx = min(atomx), max(atomx)
  miny, maxy = min(atomy), max(atomy)
  dw, dh = maxx-minx, maxy-miny
  sw, sh = render_size
  rh, rv = (sw - 4*atom_radius)/dw, (sh - 4*atom_radius)/dh
  scale = rv if rh > rv else rh
  translate = sw/2 - (maxx+minx)/2*scale, sh/2 - (maxy+miny)/2*scale, 0

  root = ET.Element('svg')
  root.set('xmlns', svg_ns)
  root.set('width',  str(sw))
  root.set('height', str(sh))
  create_defs(root)
  g = ET.SubElement(root, 'g')
  g.set('transform', 'matrix(1,0,0,-1,0,%d)' % sh)
  create_atoms(g, atoms, conns, scale, translate)
  tree = ET.ElementTree(root)
  tree.write(filename_svg, xml_declaration=True)


if __name__ == '__main__':
  render()
