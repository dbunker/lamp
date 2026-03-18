#constant(obj, o0).
#constant(obj, o3).
#constant(obj, o6).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d1(var(obj))).
#modeh(d2(var(obj))).
#modeh(d4(var(obj))).
#modeh(d5(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d1(o9), d2(o3), d2(o6), d4(o3), d4(o6), d5(o3), d5(o6), d5(o9)
}, {
  d1(o3), d1(o6), d2(o0), d2(o8), d2(o9), d4(o0), d4(o8), d5(o0), d5(o8)
}, {
  d0(o9).
  d1(o0).
  d1(o8).
  d3(o3).
  d3(o6).
  d4(o9).
}).