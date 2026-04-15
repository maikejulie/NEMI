# Writing How-to Guides

## What is a How-to Guide?

A how-to guide is **goal-oriented** documentation that helps a competent user achieve a specific real-world task or solve a particular problem.

**Key characteristics:**
- Problem-solving focused
- Task-oriented
- Assumes user competence
- Provides practical directions
- Goal-driven

**User context:**
- Has working knowledge of the tool
- Working mode (not learning)
- Has a specific problem to solve
- Needs efficient solution

**User question:** "How do I...?"

## Purpose and Goals

### Primary Purpose
Help an already-competent user accomplish a specific real-world goal efficiently.

### Success Criteria
- User achieves their specific goal
- Solution is practical and actionable
- User can adapt approach to similar problems
- Time to solution is minimized
- Steps are clear and direct

### What How-to Guides Are NOT
- Not tutorials (those teach basics to beginners)
- Not reference documentation (those describe all features)
- Not explanatory (those explain concepts and "why")
- Not comprehensive (focused on one specific goal)
- Not for beginners (assumes working knowledge)

## Core Principles

### 1. Focus on the Goal

Start with the user's goal, not the tool's features.

**Good:**
```markdown
# How to Deploy Your Application to Production

This guide shows you how to deploy your application
to a production server with zero downtime.
```

**Bad:**
```markdown
# The Deployment System

Our deployment system has many features including
blue-green deployments, canary releases, and rollback
capabilities. Let's explore each one...
```

### 2. Assume Competence

The user knows the basics. Don't teach fundamentals.

**Good:**
```markdown
# How to Add Caching

Add a Redis cache layer to improve performance:

1. Install Redis: `npm install redis`
2. Configure the connection in `config/cache.js`
3. Wrap your database queries with cache calls
```

**Bad:**
```markdown
# How to Add Caching

First, let's understand what caching is. Caching is a
technique where you store frequently accessed data
in memory to avoid repeated expensive operations...
```

### 3. Be Practical and Direct

Provide concrete steps for the specific goal, not general advice.

**Good:**
```markdown
Edit your nginx config at `/etc/nginx/sites-available/default`:
```nginx
location / {
    proxy_pass http://localhost:3000;
}
```
```

**Bad:**
```markdown
You'll need to configure your reverse proxy. Depending on
your setup, you might be using nginx, Apache, or HAProxy.
Each has different configuration approaches...
```

### 4. Acknowledge Real-World Constraints

Address actual conditions users face, including limitations and trade-offs.

**Good:**
```markdown
# How to Migrate Data with Zero Downtime

**Note**: This approach requires 2x database storage during
migration. If storage is limited, see [alternative migration
strategies](link).
```

**Bad:**
```markdown
# How to Migrate Data

Just run the migration script.
```

### 5. Provide Context for Decisions

When choices matter, briefly explain why you're recommending a particular approach.

**Good:**
```markdown
Use environment variables for sensitive configuration:
```bash
export API_KEY=your_key_here
```

This keeps secrets out of version control and allows
different values per environment.
```

**Bad:**
```markdown
Set your API key however you prefer.
```

### 6. Link to Related Resources

Don't explain concepts or provide comprehensive reference. Link instead.

**Good:**
```markdown
Configure CORS headers in your middleware:
```javascript
app.use(cors({ origin: 'https://example.com' }))
```

See [CORS options reference](link) for all available settings.
```

**Bad:**
```markdown
CORS (Cross-Origin Resource Sharing) is a security feature
that restricts resources from being accessed by web pages
from different domains. It works by using HTTP headers
that tell browsers whether a particular request...
```

## Structure of a How-to Guide

### 1. Title: "How to [Achieve Goal]"

Make the goal immediately clear.

**Good:**
- "How to Set Up Continuous Deployment"
- "How to Add OAuth Authentication"
- "How to Optimize Database Queries"

**Bad:**
- "Continuous Deployment" (not action-oriented)
- "OAuth Guide" (too vague)
- "Database Performance" (not specific)

### 2. Introduction (Brief)

**What to include:**
- What problem this solves
- What the user will achieve
- Any prerequisites or requirements
- (Optional) When this approach is appropriate

**Keep it short:** 2-3 sentences max.

**Example:**
```markdown
# How to Set Up Continuous Deployment

This guide shows you how to automatically deploy your application
to production when you push to the main branch.

**Prerequisites**: GitHub repository, production server with SSH access
```

### 3. Steps (The Main Content)

**Structure each step as:**
1. Action to take
2. Specific commands/code
3. Brief explanation if needed
4. Verification (when helpful)

**Example:**
```markdown
## Step 1: Configure GitHub Actions

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: npm install
      - run: npm run build
```

This workflow triggers on every push to main and
builds your application.

## Step 2: Add Deploy Script

Create `deploy.sh`:
```bash
#!/bin/bash
scp -r build/* user@server:/var/www/app/
```

Make it executable: `chmod +x deploy.sh`
```

### 4. Verification (Optional)

Show how to confirm the solution works.

**Example:**
```markdown
## Verify the Deployment

Push a change to the main branch:
```bash
git push origin main
```

Check the Actions tab in GitHub. You should see
a successful deployment run.

Visit your production URL to see the changes live.
```

### 5. Troubleshooting (Optional)

Address common issues specific to this task.

**Example:**
```markdown
## Troubleshooting

**Problem**: "Permission denied" error during deployment
**Solution**: Add your deployment key to the server's authorized_keys

**Problem**: Changes not appearing on production
**Solution**: Check that the build directory is being copied correctly
```

### 6. Next Steps / Related Guides (Optional)

Link to related how-to guides or relevant explanations.

**Example:**
```markdown
## Related Guides
- [How to Set Up Staging Environments →](link)
- [How to Roll Back a Deployment →](link)
```

## Writing Style

### Focus on Action

Use active voice and imperative mood.

**Good:** "Configure the database connection"
**Bad:** "The database connection should be configured"

### Be Concise

Respect the user's time. Get to the point.

**Good:**
```markdown
Install the package:
```bash
npm install express
```
```

**Bad:**
```markdown
The next thing we need to do is install Express. Express is
a popular web framework for Node.js. You can install it using
npm, which is the package manager for Node.js. To do this,
run the following command...
```

### Use Numbered Steps for Sequential Actions

Makes the flow clear and easy to follow.

**Good:**
```markdown
1. Create the config file
2. Add your credentials
3. Restart the service
```

**Bad:**
```markdown
You'll need to create a config file, and then add your credentials,
and then restart the service.
```

### Provide Exact Commands

Don't make users guess.

**Good:** `sudo systemctl restart nginx`
**Bad:** "restart the web server"

### Be Specific About File Paths

Tell users exactly where things go.

**Good:** "Edit `/etc/postgresql/12/main/pg_hba.conf`"
**Bad:** "Edit the PostgreSQL config file"

## Best Practices

### ✓ DO

- **Start with the goal**: Make it clear what this achieves
- **Assume basic knowledge**: Don't explain fundamentals
- **Provide complete examples**: Working code, not fragments
- **Use realistic scenarios**: Real-world problems, not toy examples
- **Test your instructions**: Verify every step works
- **Address common issues**: Include troubleshooting
- **Be prescriptive**: Recommend a specific approach
- **Link to references**: For detailed options/parameters
- **Show alternative approaches**: When significantly different
- **Consider user constraints**: Time, resources, expertise

### ✗ DON'T

- **Don't teach basics**: Link to tutorials instead
- **Don't explain concepts**: Link to explanations instead
- **Don't cover everything**: Focus on the specific goal
- **Don't be vague**: Provide exact commands and paths
- **Don't assume environment**: Specify prerequisites clearly
- **Don't ignore edge cases**: Address common variations
- **Don't leave users hanging**: Provide verification steps
- **Don't duplicate reference docs**: Link to API docs instead
- **Don't wander off topic**: Stay focused on the goal

## Common Patterns

### Pattern 1: Installation and Configuration
```markdown
# How to Set Up X

1. Install X
2. Create configuration file
3. Configure for your environment
4. Start the service
5. Verify it's working
```

### Pattern 2: Migration or Upgrade
```markdown
# How to Migrate from X to Y

1. Backup your current data
2. Install Y alongside X
3. Run migration script
4. Verify data integrity
5. Switch to Y
6. Clean up old X installation
```

### Pattern 3: Problem-Solving
```markdown
# How to Fix [Specific Problem]

1. Identify the cause
2. Apply the fix
3. Verify the problem is resolved
4. Prevent future occurrences
```

### Pattern 4: Integration
```markdown
# How to Integrate X with Y

1. Install required packages
2. Configure X to connect to Y
3. Set up authentication
4. Test the integration
```

### Pattern 5: Optimization
```markdown
# How to Improve [Performance Metric]

1. Identify the bottleneck
2. Apply optimization technique
3. Measure the improvement
4. Further optimizations (optional)
```

## Example: Good How-to Guide

```markdown
# How to Add Rate Limiting to Your API

Protect your API from abuse by limiting request rates per user.

**Prerequisites**: Express.js application, Redis server

## Install Dependencies

```bash
npm install express-rate-limit redis
```

## Configure Redis Connection

Create `config/redis.js`:
```javascript
const redis = require('redis');
const client = redis.createClient({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379
});

module.exports = client;
```

## Add Rate Limiting Middleware

Create `middleware/rateLimiter.js`:
```javascript
const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');
const redis = require('../config/redis');

const limiter = rateLimit({
  store: new RedisStore({ client: redis }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});

module.exports = limiter;
```

## Apply to Your Routes

In `app.js`:
```javascript
const rateLimiter = require('./middleware/rateLimiter');

// Apply to all routes
app.use(rateLimiter);

// Or apply to specific routes only
app.use('/api/', rateLimiter);
```

## Test the Rate Limit

Use curl to test:
```bash
# Make 101 requests rapidly
for i in {1..101}; do curl http://localhost:3000/api/test; done
```

After 100 requests, you should see:
```json
{"error": "Too many requests, please try again later."}
```

## Customize for Different Endpoints

Apply stricter limits to sensitive endpoints:
```javascript
const strictLimiter = rateLimit({
  store: new RedisStore({ client: redis }),
  windowMs: 15 * 60 * 1000,
  max: 5 // only 5 requests per 15 minutes
});

app.use('/api/admin', strictLimiter);
```

## Troubleshooting

**Problem**: Rate limiting not working
**Solution**: Ensure Redis is running: `redis-cli ping` (should return "PONG")

**Problem**: Different users sharing rate limits
**Solution**: Check that your load balancer is forwarding the correct IP address

## Related Guides
- [How to Set Up Redis Cluster →](link)
- [How to Monitor API Usage →](link)
```

## Checklist for How-to Guide Quality

Before publishing, verify:

- [ ] Title clearly states the goal
- [ ] Introduction identifies what problem this solves
- [ ] Prerequisites are explicitly stated
- [ ] Assumes appropriate level of user knowledge
- [ ] Steps are concrete and actionable
- [ ] All commands/code are complete and correct
- [ ] File paths and locations are specific
- [ ] Verification steps are provided
- [ ] Common issues are addressed
- [ ] Stays focused on the specific goal
- [ ] Links to reference docs instead of duplicating them
- [ ] Links to explanations instead of explaining concepts
- [ ] Tested on a clean environment

## Common Mistakes

### Mistake: Teaching Instead of Guiding
**Bad:** "Before we begin, let's understand what rate limiting is..."
**Good:** "Protect your API from abuse by limiting request rates."

### Mistake: Being Too General
**Bad:** "Configure your settings appropriately"
**Good:** "Set `max: 100` in the rate limiter config"

### Mistake: Explaining the Obvious
**Bad:** "Save the file. Saving the file writes it to disk so you can..."
**Good:** "Save the file."

### Mistake: Duplicating Reference Docs
**Bad:** "The rate limiter has these options: windowMs (number), max (number), message (string), statusCode (number)..."
**Good:** "Set `windowMs: 900000` (15 minutes). [See all options →](link)"

### Mistake: Ignoring Context
**Bad:** "Just use this configuration" (works locally but not in production)
**Good:** "For production, set these environment variables:"

### Mistake: No Verification
**Bad:** "Run the script. Done!"
**Good:** "Run the script. You should see: `Migration completed: 1000 records processed`"

---

**Remember**: A how-to guide helps someone accomplish a goal. Be practical, be specific, be direct. The user knows what they want to do—just show them how to do it efficiently.
