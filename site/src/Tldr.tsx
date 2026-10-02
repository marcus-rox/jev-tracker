import { Fragment, type ReactNode } from 'react'

const HEADING = /^(Summary|Key Points|Table|Interesting Notes):\s*$/
const SEPARATOR = /^\|[\s:|-]+\|$/
const NUMERIC = /^-?\d+(\.\d+)?$/

type Block =
  | { kind: 'h'; text: string }
  | { kind: 'p'; text: string }
  | { kind: 'ul'; items: string[] }
  | { kind: 'table'; rows: string[][] }

const cells = (line: string) => line.slice(1, -1).split('|').map((c) => c.trim())

/** Parse the recap-format markdown the daily run writes: four `Heading:` lines, bullets, one pipe table. */
function parse(text: string): Block[] {
  const blocks: Block[] = []
  const last = () => blocks[blocks.length - 1]
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line) continue
    if (HEADING.test(line)) blocks.push({ kind: 'h', text: line.slice(0, -1) })
    else if (line.startsWith('- ')) {
      const prev = last()
      if (prev?.kind === 'ul') prev.items.push(line.slice(2))
      else blocks.push({ kind: 'ul', items: [line.slice(2)] })
    } else if (line.startsWith('|') && line.endsWith('|')) {
      if (SEPARATOR.test(line)) continue
      const prev = last()
      if (prev?.kind === 'table') prev.rows.push(cells(line))
      else blocks.push({ kind: 'table', rows: [cells(line)] })
    } else blocks.push({ kind: 'p', text: line })
  }
  return blocks
}

function Cell({ tag, text }: { tag: 'th' | 'td'; text: string }) {
  const Tag = tag
  return <Tag className={NUMERIC.test(text) ? 'n' : undefined}>{text}</Tag>
}

function render(block: Block, i: number): ReactNode {
  switch (block.kind) {
    case 'h':
      return <h3 key={i}>{block.text}</h3>
    case 'p':
      return <p key={i}>{block.text}</p>
    case 'ul':
      return <ul key={i}>{block.items.map((item, j) => <li key={j}>{item}</li>)}</ul>
    case 'table': {
      const [head, ...body] = block.rows
      return (
        <table key={i}>
          <thead><tr>{head.map((c, j) => <Cell key={j} tag="th" text={c} />)}</tr></thead>
          <tbody>{body.map((row, j) => <tr key={j}>{row.map((c, k) => <Cell key={k} tag="td" text={c} />)}</tr>)}</tbody>
        </table>
      )
    }
  }
}

/** Each heading and the blocks under it form one section, so the layout can place sections in columns. */
export default function Tldr({ text }: { text: string }) {
  const sections: Block[][] = []
  for (const block of parse(text)) {
    if (block.kind === 'h' || sections.length === 0) sections.push([])
    sections[sections.length - 1].push(block)
  }
  return (
    <div className="tldr">
      {sections.map((blocks, i) => <section key={i}>{blocks.map((b, j) => <Fragment key={j}>{render(b, j)}</Fragment>)}</section>)}
    </div>
  )
}
