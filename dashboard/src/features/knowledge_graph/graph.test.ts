import { describe, expect, test } from 'bun:test'
import { edgeSize, nodeDisplayName } from './graph'

describe('edgeSize', () => {
  const range = { min: 1, max: 5 }
  test('weight 2 is only slightly thicker than weight 1', () => {
    expect(edgeSize(1, range)).toBe(1)
    expect(edgeSize(2, range)).toBe(1.5)
  })
  test('clamps to range', () => {
    expect(edgeSize(0, range)).toBe(1)
    expect(edgeSize(1e9, range)).toBe(5)
  })
})

describe('nodeDisplayName', () => {
  const node = { id: '4:abc:1', labels: ['iccn_dwh.fact_berita'], properties: {} }
  test('shows the business name the schema pipeline set', () => {
    expect(nodeDisplayName({ ...node, properties: { display_name: 'Berita' } })).toBe('Berita')
  })
  test('falls back to the entity name, then the id', () => {
    expect(nodeDisplayName(node)).toBe('iccn_dwh.fact_berita')
    expect(nodeDisplayName({ ...node, labels: [], properties: { display_name: '' } })).toBe('4:abc:1')
  })
})
