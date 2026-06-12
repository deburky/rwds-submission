# RWDS project conventions

## Comment style for section dividers

Use the 80-character dashed style — not Unicode box-drawing characters:

```python
# -------------------------------------------------------------------------------
# Section name
# -------------------------------------------------------------------------------
```

No blank line between the closing `#---` line and the next code block.

Do NOT use: `# ── Section ──────────────────────────────────────────────────────`

## Assignment formatting

No aligned `=` signs. One space on each side only:

```python
# correct
foo = "bar"
longer_name = "baz"

# wrong
foo         = "bar"
longer_name = "baz"
```

## F-string formatting

No extra spaces inside f-strings around interpolated values:

```python
# correct
f"value={var}"
f"{a} and {b}"

# wrong
f"value=   {var}"
f"    {var}"
f"{var} "
```