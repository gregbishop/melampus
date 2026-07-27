--[[
Minimal JSON decoder.

The Lightroom SDK ships no JSON library, so this exists rather than adding a
dependency. Decode only — nothing here needs to write JSON.

Written against Lua 5.1, which is what Lightroom runs. Avoids goto, integer
division, and bitwise operators for that reason.

Returns `nil, message` on malformed input rather than raising, because a
truncated results file must produce a clear error in the UI, not a stack trace.
--]]

local Json = {}

local escapes = {
	['"'] = '"', ['\\'] = '\\', ['/'] = '/',
	b = '\b', f = '\f', n = '\n', r = '\r', t = '\t',
}

-- Encode a Unicode code point as UTF-8. Lightroom's Lua has no utf8 library.
local function utf8_encode(code)
	if code < 0x80 then
		return string.char(code)
	elseif code < 0x800 then
		return string.char(0xC0 + math.floor(code / 0x40), 0x80 + (code % 0x40))
	elseif code < 0x10000 then
		return string.char(
			0xE0 + math.floor(code / 0x1000),
			0x80 + (math.floor(code / 0x40) % 0x40),
			0x80 + (code % 0x40))
	end
	return string.char(
		0xF0 + math.floor(code / 0x40000),
		0x80 + (math.floor(code / 0x1000) % 0x40),
		0x80 + (math.floor(code / 0x40) % 0x40),
		0x80 + (code % 0x40))
end

local parse_value

local function skip_space(text, pos)
	local _, stop = string.find(text, '^[ \t\r\n]*', pos)
	return stop + 1
end

local function parse_string(text, pos)
	pos = pos + 1 -- opening quote
	local pieces = {}
	while true do
		local chunk_start, chunk_stop = string.find(text, '^[^"\\]*', pos)
		if chunk_stop >= chunk_start then
			pieces[#pieces + 1] = string.sub(text, chunk_start, chunk_stop)
			pos = chunk_stop + 1
		end
		local char = string.sub(text, pos, pos)
		if char == '' then
			return nil, 'unterminated string'
		elseif char == '"' then
			return table.concat(pieces), pos + 1
		end
		-- backslash escape
		local esc = string.sub(text, pos + 1, pos + 1)
		if esc == 'u' then
			local hex = string.sub(text, pos + 2, pos + 5)
			local code = tonumber(hex, 16)
			if not code then return nil, 'bad \\u escape' end
			pos = pos + 6
			-- surrogate pair
			if code >= 0xD800 and code <= 0xDBFF and string.sub(text, pos, pos + 1) == '\\u' then
				local low = tonumber(string.sub(text, pos + 2, pos + 5), 16)
				if low and low >= 0xDC00 and low <= 0xDFFF then
					code = 0x10000 + (code - 0xD800) * 0x400 + (low - 0xDC00)
					pos = pos + 6
				end
			end
			pieces[#pieces + 1] = utf8_encode(code)
		else
			local mapped = escapes[esc]
			if not mapped then return nil, 'bad escape \\' .. esc end
			pieces[#pieces + 1] = mapped
			pos = pos + 2
		end
	end
end

local function parse_array(text, pos)
	local out = {}
	pos = skip_space(text, pos + 1)
	if string.sub(text, pos, pos) == ']' then return out, pos + 1 end
	while true do
		local value, err = nil, nil
		value, pos, err = parse_value(text, pos)
		if value == nil and err then return nil, nil, err end
		out[#out + 1] = value
		pos = skip_space(text, pos)
		local char = string.sub(text, pos, pos)
		if char == ',' then
			pos = skip_space(text, pos + 1)
		elseif char == ']' then
			return out, pos + 1
		else
			return nil, nil, 'expected , or ] in array'
		end
	end
end

local function parse_object(text, pos)
	local out = {}
	pos = skip_space(text, pos + 1)
	if string.sub(text, pos, pos) == '}' then return out, pos + 1 end
	while true do
		if string.sub(text, pos, pos) ~= '"' then
			return nil, nil, 'expected string key in object'
		end
		local key, err = parse_string(text, pos)
		if key == nil then return nil, nil, err end
		pos = err -- parse_string returns (value, nextPos)
		pos = skip_space(text, pos)
		if string.sub(text, pos, pos) ~= ':' then
			return nil, nil, 'expected : after key'
		end
		pos = skip_space(text, pos + 1)
		local value
		value, pos, err = parse_value(text, pos)
		if value == nil and err then return nil, nil, err end
		out[key] = value
		pos = skip_space(text, pos)
		local char = string.sub(text, pos, pos)
		if char == ',' then
			pos = skip_space(text, pos + 1)
		elseif char == '}' then
			return out, pos + 1
		else
			return nil, nil, 'expected , or } in object'
		end
	end
end

-- JSON null has no Lua equivalent that survives table storage, so it becomes a
-- sentinel the caller can test with Json.isNull rather than vanishing silently.
Json.null = setmetatable({}, { __tostring = function() return 'null' end })

function Json.isNull(value)
	return value == Json.null
end

parse_value = function(text, pos)
	pos = skip_space(text, pos)
	local char = string.sub(text, pos, pos)
	if char == '"' then
		local value, nextPos = parse_string(text, pos)
		if value == nil then return nil, nil, nextPos end
		return value, nextPos
	elseif char == '{' then
		return parse_object(text, pos)
	elseif char == '[' then
		return parse_array(text, pos)
	elseif char == '' then
		return nil, nil, 'unexpected end of input'
	end

	local literal = string.sub(text, pos, pos + 4)
	if string.sub(literal, 1, 4) == 'true' then return true, pos + 4 end
	if string.sub(literal, 1, 5) == 'false' then return false, pos + 5 end
	if string.sub(literal, 1, 4) == 'null' then return Json.null, pos + 4 end

	local numText = string.match(text, '^%-?%d+%.?%d*[eE]?[%+%-]?%d*', pos)
	if numText and numText ~= '' then
		local number = tonumber(numText)
		if number then return number, pos + string.len(numText) end
	end
	return nil, nil, 'unexpected character "' .. char .. '"'
end

--- Decode a JSON document. Returns the value, or nil plus a message.
function Json.decode(text)
	if type(text) ~= 'string' or text == '' then
		return nil, 'empty input'
	end
	local value, pos, err = parse_value(text, 1)
	if value == nil and err then return nil, err end
	pos = skip_space(text, pos)
	if pos <= string.len(text) then
		return nil, 'trailing content at position ' .. tostring(pos)
	end
	return value
end

return Json
