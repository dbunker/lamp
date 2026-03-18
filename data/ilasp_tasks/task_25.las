#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o3).
#constant(obj, o5).

#modeh(d2(var(obj))).
#modeh(d8(var(obj))).

#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d2(o5), d8(o1)
}, {
  d2(o3), d8(o0)
}, {
  d2(o0).
  d2(o1).
  d3(o5).
  d4(o1).
  d4(o5).
  d5(o1).
  d8(o3).
  d8(o5).
  d9(o5).
}).