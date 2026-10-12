import { describe, expect, test } from 'bun:test'
import { edgeSize } from './graph'

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
