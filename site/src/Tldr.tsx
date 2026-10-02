import type { ReactNode } from 'react'

const BULLET = /^(\s*)- (.*)$/
const INDENT_SPACES = 2

type Node = { text: string; children: Node[] }

/** Parse the nested-bullet markdown the daily run writes; a line without a dash is a paragraph (depth -1). */
function parse(text: string): { paragraphs: string[]; roots: Node[] } {
  const paragraphs: string[] = []
  const roots: Node[] = []
  const stack: Node[][] = [roots]
  for (const raw of text.split('\n')) {
    if (!raw.trim()) continue
    const match = BULLET.exec(raw)
    if (!match) {
      paragraphs.push(raw.trim())
      continue
    }
    const depth = Math.min(Math.floor(match[1].length / INDENT_SPACES), stack.length - 1)
    const node: Node = { text: match[2].trim(), children: [] }
    stack.length = depth + 1
    stack[depth].push(node)
    stack.push(node.children)
  }
  return { paragraphs, roots }
}

function List({ nodes }: { nodes: Node[] }): ReactNode {
  return (
    <ul>
      {nodes.map((node, i) => (
        <li key={i}>
          {node.text}
          {node.children.length > 0 && <List nodes={node.children} />}
        </li>
      ))}
    </ul>
  )
}

export default function Tldr({ text }: { text: string }) {
  const { paragraphs, roots } = parse(text)
  return (
    <div className="tldr">
      {paragraphs.map((line, i) => <p key={i}>{line}</p>)}
      {roots.length > 0 && <List nodes={roots} />}
    </div>
  )
}
