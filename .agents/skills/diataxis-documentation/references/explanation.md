# Writing Explanations

## What is an Explanation?

Explanation is **understanding-oriented** documentation that provides context, background, and clarifies concepts to deepen the user's understanding of a subject.

**Key characteristics:**
- Understanding-oriented
- Provides context and background
- Explains "why" and "how it works"
- Discusses alternatives and trade-offs
- Can include opinions and perspectives

**User context:**
- Learning and building understanding
- Study mode (not actively working)
- Wants to understand concepts
- Seeking deeper knowledge

**User question:** "Why...?" or "How does this work?" or "What is...?"

## Purpose and Goals

### Primary Purpose
Help users understand concepts, design decisions, and how things work at a deeper level.

### Success Criteria
- User gains understanding of concepts
- User understands "why" things are the way they are
- User can make informed decisions
- User sees the bigger picture
- User understands trade-offs and alternatives

### What Explanations Are NOT
- Not step-by-step instructions (that's Tutorials or How-to Guides)
- Not technical specifications (that's Reference)
- Not problem-solving (that's How-to Guides)
- Not meant for quick lookups (that's Reference)
- Not teaching through doing (that's Tutorials)

## Core Principles

### 1. Provide Context and Background

Explain the "why" behind decisions and the context around concepts.

**Good:**
```markdown
# Why We Use Event-Driven Architecture

We chose event-driven architecture for this system because
our domain involves multiple independent services that need
to react to state changes without tight coupling.

Traditional request-response patterns would create dependencies
between services, making the system harder to scale and maintain.
With events, each service can evolve independently.

However, this approach introduces eventual consistency, which
means there's a brief window where different services may have
different views of the data. This trade-off is acceptable for
our use case because...
```

**Bad:**
```markdown
# Event-Driven Architecture

To implement events, follow these steps...
(This is a how-to, not an explanation)
```

### 2. Make Connections

Link concepts together and show how they relate to each other and the bigger picture.

**Good:**
```markdown
# Understanding Authentication and Authorization

Authentication and authorization are often confused, but they
serve distinct purposes in security:

**Authentication** answers "who are you?" It verifies identity.

**Authorization** answers "what can you do?" It controls access.

They work together in a typical flow:
1. User authenticates (proves identity)
2. System looks up user's roles/permissions
3. System authorizes (or denies) each requested action

This separation allows you to change authorization rules
without re-authenticating users, and to authenticate users
without knowing in advance what they'll try to access.
```

**Bad:**
```markdown
# Authentication

Call the `authenticate()` function with credentials...
(This is reference or how-to, not explanation)
```

### 3. Discuss Alternatives and Trade-offs

Explore different approaches, explaining pros and cons.

**Good:**
```markdown
# Database Choice: SQL vs. NoSQL

For this application, we considered both SQL and NoSQL databases.

**SQL (PostgreSQL) Advantages:**
- ACID transactions ensure data consistency
- Mature ecosystem and tooling
- Powerful query capabilities with JOINs
- Well-understood by the team

**SQL Disadvantages:**
- Schema changes can be complex
- Horizontal scaling is more difficult
- Fixed schema doesn't fit all data

**NoSQL (MongoDB) Advantages:**
- Flexible schema fits varied data
- Easier horizontal scaling
- Better performance for certain access patterns

**NoSQL Disadvantages:**
- No transactions across documents (in older versions)
- Less mature tooling
- Eventual consistency can be complex

We chose PostgreSQL because our data is highly relational
and consistency is critical for financial transactions.
```

**Bad:**
```markdown
# Database Setup

Install PostgreSQL:
```bash
apt-get install postgresql
```
(This is a tutorial/how-to, not explanation)
```

### 4. Explain "How It Works"

Describe mechanisms, processes, and internal workings at a conceptual level.

**Good:**
```markdown
# How Caching Improves Performance

When you request data from the database, it's a slow operation:
1. Network round-trip to database server
2. Database query execution
3. Data serialization and transfer

A cache sits between your application and the database,
storing frequently-accessed data in memory. When you request
data:

**First request (cache miss):**
1. Check cache → not found
2. Query database
3. Store result in cache
4. Return result
**Time: ~50ms**

**Subsequent requests (cache hit):**
1. Check cache → found
2. Return cached result
**Time: ~1ms**

This 50x speedup comes with trade-offs:
- **Staleness**: Cached data may be outdated
- **Memory usage**: Cache consumes RAM
- **Complexity**: Cache invalidation is notoriously difficult

The benefit is usually worth these costs for read-heavy
workloads where data doesn't change frequently.
```

**Bad:**
```markdown
# Caching

Caching stores data in memory for fast access. It's important
for performance. Here's how to implement it...
(Too brief and moves into how-to)
```

### 5. Accommodate Perspective and Opinion

Unlike reference documentation, explanations can include opinions, perspectives, and subjective insights.

**Good:**
```markdown
# Our Microservices Philosophy

We believe microservices should be organized around business
capabilities, not technical layers. This is somewhat controversial
in the industry, where many teams organize services by data types
or technical functions.

We've found that business-capability boundaries are more stable
over time. Product features change, but core business capabilities
("process orders," "manage inventory") remain consistent.

This approach has served us well, though it does mean some
code duplication across services. We consider this acceptable—
it's a trade-off we've consciously chosen for better service
independence.
```

**Bad:**
```markdown
# Microservices

Microservices are an architectural style where applications
are composed of small, independent services. Each service...
(Too dry, sounds like reference material)
```

### 6. Go Deeper Than Necessary

Explanations can explore topics beyond immediate practical needs, satisfying curiosity.

**Good:**
```markdown
# The History of Our API Design

Our API started as a traditional REST API in 2018. As we grew,
we hit limitations:
- Mobile apps needed different data shapes than web
- Multiple round-trips caused slow performance
- Version management became complex

We evaluated several alternatives:
- **REST with better batching**: Didn't solve different-data-shapes problem
- **gRPC**: Better performance but poor browser support
- **GraphQL**: Solved data shape and round-trip issues

We chose GraphQL in 2020. The migration took six months...

Today, we're exploring edge computing with GraphQL resolvers
at the CDN level, which would reduce latency further...
```

This historical context and future thinking isn't immediately practical, but helps readers understand the evolution and direction.

## Structure of an Explanation

### 1. Title and Introduction

State what you're explaining and why it matters.

**Example:**
```markdown
# Understanding Eventual Consistency

Eventual consistency is a key concept in distributed systems that
often confuses developers coming from traditional database backgrounds.
Understanding it is crucial for working with distributed architectures.
```

### 2. Main Content (Flexible Structure)

Unlike other documentation types, explanations don't follow a rigid structure. Common approaches:

**Concept Introduction:**
- Define the concept
- Explain why it exists
- Show how it relates to other concepts

**Historical/Evolutionary:**
- How did we get here?
- What problems were we trying to solve?
- How has thinking evolved?

**Comparative:**
- Compare different approaches
- Explain trade-offs
- When to use each

**Mechanism Explanation:**
- How does this work internally?
- What are the components?
- How do they interact?

**Problem/Solution:**
- What problem does this solve?
- Why is it solved this way?
- What are the implications?

### 3. Conclusion (Optional)

Summarize key points or provide perspective.

**Example:**
```markdown
## Key Takeaways

Eventual consistency is a trade-off: you gain availability
and partition tolerance at the cost of immediate consistency.
For many modern applications, this trade-off makes sense,
but it requires different thinking about data and operations.
```

## Writing Style

### Use Conversational Tone

Explanations can be more conversational than other documentation types.

**Good:** "Let's explore why we made this decision..."
**Acceptable:** "You might wonder why..."
**Acceptable:** "This is a common point of confusion..."

### Tell a Story

Narrative structure helps understanding.

**Good:**
```markdown
When we first built the notification system, we used a simple
polling approach. Every 30 seconds, clients would ask: "any
new notifications?" This worked fine with 100 users.

At 10,000 users, we had a problem. The server was handling
20,000 requests per minute just for polling, and most returned
"no new notifications." We were wasting resources.

This led us to WebSockets...
```

**Bad:**
```markdown
Notification systems can use polling or WebSockets. Each has
advantages and disadvantages. Polling is simpler. WebSockets
are more efficient.
```

### Use Analogies and Metaphors

Help readers build mental models.

**Good:**
```markdown
Think of a database transaction like a shopping cart at a store.
You can add items (changes) to your cart, but nothing is final
until you check out (commit). If you change your mind, you can
put everything back (rollback) and it's as if you never picked
up those items.
```

### Explore "What If" Scenarios

Help readers understand implications.

**Good:**
```markdown
What if we didn't use a load balancer?

With no load balancer, you'd need to:
- Give clients the list of all server addresses
- Implement client-side load distribution logic
- Handle server failures in every client
- Update all clients when servers change

This complexity in every client is why we centralize it
in a load balancer.
```

### Include Diagrams and Visuals

Explanations benefit greatly from diagrams.

**Good:**
```markdown
# How Request Routing Works

```
Client → Load Balancer → Server 1
                       → Server 2
                       → Server 3
```

The load balancer receives all client requests and distributes
them across available servers based on current load...
```

## Best Practices

### ✓ DO

- **Explain "why"**: The reasoning behind decisions
- **Provide context**: Historical, technical, business
- **Make connections**: Link concepts to the bigger picture
- **Discuss alternatives**: What else could we have done?
- **Share trade-offs**: What did we gain? What did we sacrifice?
- **Use examples**: Concrete scenarios to illustrate abstract concepts
- **Include opinions**: Perspectives and subjective insights
- **Tell stories**: Narrative helps understanding
- **Go deep**: Satisfy curiosity beyond immediate needs
- **Admit uncertainty**: "We're still learning..." is OK
- **Link to other docs**: Reference specs, tutorials, how-tos

### ✗ DON'T

- **Don't provide step-by-step instructions**: That's tutorials/how-tos
- **Don't try to be comprehensive reference**: Link to reference instead
- **Don't solve specific problems**: That's how-to guides
- **Don't assume no prior knowledge**: Explanations build on basics
- **Don't be dogmatic**: Acknowledge different valid perspectives
- **Don't ignore complexity**: Simplify, but don't oversimplify
- **Don't forget the "why"**: Explanation without "why" is just description

## Common Patterns

### Pattern 1: Concept Explanation
```markdown
# Understanding [Concept]

## What is [Concept]?
Definition and basic understanding

## Why [Concept] Exists
Problems it solves, context for its existence

## How [Concept] Works
Mechanisms and processes

## When to Use [Concept]
Situations where it's appropriate

## Trade-offs and Alternatives
What else exists and why choose this
```

### Pattern 2: Design Decision Explanation
```markdown
# Why We Chose [Technology/Approach]

## The Problem We Faced
Context and requirements

## Alternatives We Considered
Options and their pros/cons

## Why We Chose [This]
Reasoning for the decision

## What We Learned
Results and reflections
```

### Pattern 3: How It Works
```markdown
# How [System/Feature] Works

## Overview
High-level description

## Components
Key parts and their roles

## Flow
How it all works together

## Edge Cases
Interesting scenarios and how they're handled
```

### Pattern 4: Comparative Explanation
```markdown
# [Approach A] vs [Approach B]

## Overview of Each
Brief description of both

## Key Differences
Where they diverge

## When to Use Each
Guidance on choosing

## Our Choice and Why
If applicable, what we chose
```

## Example: Good Explanation

```markdown
# Understanding Our Caching Strategy

## The Performance Challenge

Our application serves product data that changes infrequently
(maybe once per day) but is accessed thousands of times per
second. Without caching, every request hits the database:

- Database queries: ~50ms per request
- At 1000 requests/second: 50,000ms of DB time per second
- Database becomes the bottleneck
- Response time suffers

## The Caching Approach

We cache product data in Redis, a fast in-memory store:

**First request (cache miss):**
```
Client → API → Redis (miss) → Database → Redis (store) → API → Client
Total: ~50ms
```

**Subsequent requests (cache hit):**
```
Client → API → Redis (hit) → API → Client
Total: ~2ms
```

This 25x speedup means we can handle 1000 requests/second
using only 2 seconds of total processing time.

## The Staleness Trade-off

Caching introduces a problem: **stale data**. When a product
price changes in the database, cached copies don't automatically
update.

We considered several approaches:

### Approach 1: Time-Based Expiration
Cache entries expire after a fixed time (e.g., 1 hour).

**Pros:**
- Simple to implement
- Predictable memory usage

**Cons:**
- Data can be stale for up to 1 hour
- Cache misses happen periodically even for popular items

### Approach 2: Manual Invalidation
Explicitly delete cache entries when data changes.

**Pros:**
- Cache is always current
- No unnecessary invalidation

**Cons:**
- Must remember to invalidate on every data change
- Easy to miss edge cases
- Doesn't handle external data changes

### Approach 3: Event-Based Invalidation
Database triggers emit events when data changes; cache listens
and invalidates automatically.

**Pros:**
- Automatic and reliable
- Catches all changes

**Cons:**
- Complex to set up
- Adds infrastructure (event bus)
- Possible race conditions

## Our Choice: Hybrid Approach

We use **time-based expiration (5 minutes) + manual invalidation**:

- Default 5-minute TTL means data is never more than 5 minutes stale
- Critical operations (price changes, inventory updates) manually invalidate cache
- Balances simplicity with freshness

This isn't perfect. We still have brief staleness windows, but
they're acceptable for our use case. An e-commerce site might
need event-based invalidation for inventory; we sell less
time-sensitive content.

## The Cache Warming Problem

With time-based expiration, popular items expire periodically.
When they do, the next request is slow (cache miss).

We solve this with **background refresh**: 30 seconds before
expiration, we refresh the cache in the background. Users
never experience cache misses for popular items.

This adds complexity, but the user experience improvement is
worth it.

## What We Learned

Caching isn't just about speed—it's about trade-offs:
- **Staleness vs. freshness**: How up-to-date must data be?
- **Complexity vs. simplicity**: Is advanced caching worth the maintenance?
- **Memory vs. database load**: What's your bottleneck?

For us, a simple TTL-based approach with selective invalidation
hits the sweet spot. Your needs might differ.

## Further Reading

- [How to implement caching →](link-to-how-to)
- [Cache API reference →](link-to-reference)
- [Getting started tutorial →](link-to-tutorial)
```

## Checklist for Explanation Quality

Before publishing, verify:

- [ ] Explains "why" not just "what"
- [ ] Provides context and background
- [ ] Discusses alternatives or trade-offs
- [ ] Makes connections to broader concepts
- [ ] Includes examples or scenarios
- [ ] Tells a coherent story
- [ ] Avoids step-by-step instructions
- [ ] Avoids comprehensive technical reference
- [ ] Links to other documentation types
- [ ] Appropriate level of depth
- [ ] Clear and engaging writing
- [ ] Diagrams or visuals where helpful

## Common Mistakes

### Mistake: Turning Into a Tutorial
**Bad:**
```markdown
# Understanding Authentication

First, install the auth package:
```bash
npm install auth
```

Then create a config file...
```

**Good:**
```markdown
# Understanding Authentication

Authentication verifies user identity through credentials.
The process typically involves...
```

### Mistake: Duplicating Reference Material
**Bad:**
```markdown
The `authenticate()` function takes parameters: username (string),
password (string), options (object)...
```

**Good:**
```markdown
Authentication typically requires credentials (username and password)
and may include additional options. [See API reference](link)
```

### Mistake: Being Too Dry
**Bad:** "Caching stores data. It's faster than databases."
**Good:** "Imagine if every time you wanted to check your phone number,
you had to look it up in a phone book. You'd probably remember it
after the first lookup. That's caching."

### Mistake: No "Why"
**Bad:** "We use microservices. Each service is independent."
**Good:** "We chose microservices because our teams were blocking
each other with monolith deploys. Independent services let teams
deploy independently."

### Mistake: Ignoring Trade-offs
**Bad:** "Use this approach because it's best"
**Good:** "We chose this approach because it optimizes for X at the
cost of Y, which aligns with our priorities."

### Mistake: Too Much Detail
**Bad:** Explaining every parameter, every edge case, every line of code
**Good:** Explaining concepts, mechanisms, decisions at appropriate level

---

**Remember**: Explanations help people understand. They answer "why" and "how it works" at a conceptual level. They provide context, explore alternatives, and build mental models. They can be conversational, opinionated, and deeper than immediately practical. Use them to help your users really *get* what's going on.
