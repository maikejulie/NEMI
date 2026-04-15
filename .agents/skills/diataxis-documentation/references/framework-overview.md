# Diataxis Framework Overview

This reference provides comprehensive details about the Diataxis documentation framework, its principles, and how to apply it effectively.

## What is Diataxis?

Diataxis is a systematic approach to technical documentation authoring that organizes content based on user needs. It identifies four distinct types of documentation, each serving a different purpose in the user's journey.

The name "Diataxis" comes from the ancient Greek διάταξις, meaning "arrangement" or "disposition."

## The Two-Dimensional Model

Diataxis organizes documentation along two axes:

### Axis 1: Action vs. Cognition
- **Action**: Documentation that informs practical steps (what to DO)
- **Cognition**: Documentation that informs understanding (what to KNOW)

### Axis 2: Acquisition vs. Application
- **Acquisition** (Study): User is learning and acquiring new skills
- **Application** (Work): User is applying existing knowledge to accomplish tasks

## The Four Documentation Types

| Type | Orientation | Context | Content | User Question |
|------|-------------|---------|---------|---------------|
| **Tutorial** | Learning-oriented | Study + Action | Guided lesson | "Can you teach me...?" |
| **How-to Guide** | Goal-oriented | Work + Action | Problem-solving steps | "How do I...?" |
| **Reference** | Information-oriented | Work + Cognition | Technical description | "What exactly is...?" |
| **Explanation** | Understanding-oriented | Study + Cognition | Background & context | "Why...?" |

### Visual Representation

```
                        User is STUDYING (acquiring skills)
                                    ↓

            TUTORIALS                        EXPLANATION
         Learning-oriented              Understanding-oriented
         "Teach me to..."                   "Help me understand..."
                ↓                                   ↓

User needs  → [Action]  ←━━━━━━━━━━━━━━━━━━━→ [Cognition] ← User needs
to DO                                                         to KNOW
                ↓                                   ↓

         HOW-TO GUIDES                      REFERENCE
         Goal-oriented                  Information-oriented
         "Help me achieve..."              "Tell me facts about..."

                                    ↑
                        User is WORKING (applying skills)
```

## The Compass: Identifying Documentation Type

The Diataxis compass helps you classify content by asking two key questions:

### Question 1: Does this content primarily inform ACTION or COGNITION?
- **Action**: Contains steps, instructions, procedures, commands to execute
- **Cognition**: Explains concepts, provides context, describes facts

### Question 2: Is the user ACQUIRING skills or APPLYING skills?
- **Acquiring** (Study): User is learning something new, building understanding
- **Applying** (Work): User has knowledge and is using it to accomplish something

### Decision Matrix

| Action + Acquisition | Action + Application |
|---------------------|---------------------|
| **TUTORIAL** | **HOW-TO GUIDE** |
| Learning by doing | Solving a problem |

| Cognition + Acquisition | Cognition + Application |
|------------------------|------------------------|
| **EXPLANATION** | **REFERENCE** |
| Understanding why | Looking up facts |

## Core Principles

### 1. Separation of Concerns

Each documentation type should maintain its distinct purpose. Mixing purposes creates "blur" that confuses users and reduces effectiveness.

**Examples of harmful blur:**
- Tutorials that try to explain everything (overwhelming the learner)
- Reference material that tries to teach (not useful for quick lookups)
- How-to guides that explain concepts (distracting from the goal)
- Explanations that include step-by-step instructions (loses conceptual focus)

### 2. User-Centered Design

Documentation should be organized around user needs, not product features or author preferences.

**Key considerations:**
- What is the user trying to accomplish right now?
- What stage of the journey is the user in?
- What kind of information serves this user's current need?

### 3. Iterative Improvement

Diataxis emphasizes continuous, small improvements over large restructuring efforts.

**The improvement cycle:**
1. **Choose**: Pick a small piece of documentation
2. **Assess**: Evaluate against Diataxis principles
3. **Decide**: Identify one specific improvement
4. **Do**: Make that single change and publish it

**Anti-patterns to avoid:**
- Planning entire documentation structure upfront
- Creating empty sections for all four types
- Working on large tranches of documentation at once
- Waiting for "complete" documentation before publishing

### 4. Organic Structure

Documentation structure should emerge naturally from content improvements, not be imposed externally.

**Good practice:**
- Create documentation types only when content demands it
- Let similar content naturally group together
- Move content to new sections when it makes sense
- Don't create placeholder sections

**Bad practice:**
- Creating empty "Tutorials," "How-to," "Reference," "Explanation" sections
- Forcing all documentation into the four-type structure
- Planning structure before creating content

## Maintaining Distinctness

Each documentation type has specific characteristics that keep it focused and effective:

### Tutorials: Learning-Oriented
- **DO**: Guide through complete practical lesson
- **DO**: Focus on getting user to successful result
- **DO**: Minimize explanation and choices
- **DON'T**: Explain concepts in depth
- **DON'T**: Offer alternatives or options
- **DON'T**: Assume prior knowledge beyond basics

### How-to Guides: Goal-Oriented
- **DO**: Focus on achieving a specific goal
- **DO**: Provide practical directions
- **DO**: Assume user competence
- **DON'T**: Teach concepts
- **DON'T**: Explain why unless critical
- **DON'T**: Cover unrelated topics

### Reference: Information-Oriented
- **DO**: Describe accurately and completely
- **DO**: Maintain neutral, objective tone
- **DO**: Structure according to product architecture
- **DON'T**: Include instructions or tutorials
- **DON'T**: Add opinions or recommendations
- **DON'T**: Explain concepts

### Explanation: Understanding-Oriented
- **DO**: Provide context and background
- **DO**: Explain "why" things are the way they are
- **DO**: Discuss alternatives and trade-offs
- **DON'T**: Include step-by-step instructions
- **DON'T**: Try to be comprehensive reference
- **DON'T**: Focus on one specific task

## Common Mistakes

### 1. Top-Down Planning
**Mistake**: Creating a complete four-type structure before writing content.
**Solution**: Start with existing content and improve it iteratively using Diataxis principles.

### 2. Documentation Type Confusion
**Mistake**: Tutorial that explains concepts extensively, or how-to guide that teaches basics.
**Solution**: Use the compass to identify the primary user need and maintain focus.

### 3. Trying to Do Everything
**Mistake**: One piece of documentation trying to teach, guide, reference, and explain.
**Solution**: Split into multiple pieces, each with a clear, single purpose.

### 4. Empty Structure
**Mistake**: Creating sections for all four types with "Coming soon" placeholders.
**Solution**: Create documentation only when you have content. Structure emerges from content.

### 5. Perfectionism
**Mistake**: Waiting until documentation is "complete" before publishing.
**Solution**: Publish small improvements continuously. Perfection comes through iteration.

### 6. Ignoring Existing Style
**Mistake**: Imposing Diataxis rigidly without considering project context.
**Solution**: Adapt Diataxis to fit your project's needs and existing conventions.

## Relationships Between Documentation Types

While each type should remain distinct, they can and should reference each other:

### From Tutorials
- Link to **Explanations** for users who want deeper understanding
- Link to **Reference** for detailed parameter information
- Link to **How-to Guides** for related real-world tasks

### From How-to Guides
- Link to **Tutorials** if user might lack foundational knowledge
- Link to **Reference** for technical details about commands/APIs used
- Link to **Explanations** for context on why this approach works

### From Reference
- Provide minimal examples without teaching (can link to **Tutorials**)
- Avoid explanations (link to **Explanations** instead)
- Focus solely on accurate, factual description

### From Explanations
- Reference related **Tutorials** for hands-on learning
- Link to **How-to Guides** for practical applications
- Point to **Reference** for technical specifications

## Quality Metrics

Documentation quality in Diataxis is measured by:

1. **User Alignment**: Does it serve the right user need at the right time?
2. **Focus**: Does it maintain clear purpose without blur?
3. **Completeness**: Within its type, does it provide what's needed?
4. **Accessibility**: Can users find and understand it?
5. **Accuracy**: Is the information correct and current?

## Applying Diataxis to Your Project

### Starting Point

**If you have existing documentation:**
1. Pick any piece of documentation
2. Identify its primary user need using the compass
3. Assess how well it serves that need
4. Make one small improvement aligned with its type
5. Repeat

**If you're starting from scratch:**
1. Identify what users need most urgently (often: tutorial or how-to)
2. Create that single piece of documentation
3. Iteratively add more content based on user feedback
4. Let structure emerge organically

### Common Scenarios

**Building a new tool:**
1. Start with Tutorial (help users get started)
2. Add Reference (document the API/commands)
3. Add How-to Guides (solve common problems)
4. Add Explanations (help users understand design decisions)

**Improving existing documentation:**
1. Identify pages that try to do too much
2. Split them according to Diataxis types
3. Ensure each piece has a clear, single purpose
4. Add cross-references between related pieces

## Integration with Other Systems

Diataxis is compatible with:
- **Docs-as-code**: Version control, CI/CD for documentation
- **Static site generators**: Jekyll, Hugo, Sphinx, MkDocs
- **Documentation platforms**: ReadTheDocs, GitBook, Docusaurus
- **Agile workflows**: Iterative improvement fits naturally with sprints
- **Content management systems**: Can structure content hierarchically

## Further Learning

- The Diataxis website: https://diataxis.fr/
- Focus on understanding user needs first, framework second
- Practice identifying documentation types in existing docs
- Experiment with improving one piece of documentation at a time

---

**Remember**: Diataxis is a guide, not a rigid rulebook. The goal is better documentation that serves user needs effectively. Adapt the framework to your project's context while maintaining the core principle of separation based on user needs.
