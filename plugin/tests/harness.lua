--[[ Tiny assertion harness. Lightroom's Lua has no test framework and the
     plugin must not depend on one; this is enough to drive the rules modules. ]]
local t = { passed = 0, failed = 0, failures = {} }

local function fail(msg)
	t.failed = t.failed + 1
	t.failures[#t.failures + 1] = (t.current or '?') .. ': ' .. tostring(msg)
end

function t.test(name, fn)
	t.current = name
	local ok, err = pcall(fn)
	if ok then t.passed = t.passed + 1 else fail(err) end
	t.current = nil
end

function t.isTrue(v, m) if v ~= true then error(m or 'expected true, got ' .. tostring(v), 2) end end
function t.isFalse(v, m) if v ~= false then error(m or 'expected false, got ' .. tostring(v), 2) end end
function t.isNil(v, m) if v ~= nil then error(m or 'expected nil, got ' .. tostring(v), 2) end end
function t.isNotNil(v, m) if v == nil then error(m or 'expected a value, got nil', 2) end end
function t.equals(a, b, m)
	if a ~= b then error((m or 'not equal') .. ': ' .. tostring(a) .. ' ~= ' .. tostring(b), 2) end
end
function t.contains(list, needle, m)
	for _, v in ipairs(list or {}) do if v == needle then return end end
	error((m or 'missing entry') .. ': ' .. tostring(needle), 2)
end

-- Every suite ends with `return t.summary()`; a failing one ends the
-- interpreter here with exit 1, so `lua test_x.lua` is its own gate.
function t.summary()
	for _, f in ipairs(t.failures) do print('  FAIL ' .. f) end
	print(string.format('%d passed, %d failed', t.passed, t.failed))
	if t.failed > 0 then os.exit(1) end
	return t.failed
end

return t
