#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o5).
#constant(obj, o7).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d8(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d8(o8)
}, {
  d8(o0), d8(o2), d8(o3), d8(o7), d8(o9)
}, {
  d1(o0).
  d1(o9).
  d2(o7).
  d3(o5).
  d4(o8).
  d6(o8).
  d7(o2).
  d8(o5).
  d9(o3).
}).