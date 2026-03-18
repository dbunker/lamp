#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).
#constant(obj, o6).

#modeh(d2(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).
#modeh(d5(var(obj))).
#modeh(d8(var(obj))).

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
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d2(o3), d3(o3), d4(o3), d5(o3), d8(o3)
}, {
  d2(o0), d2(o2), d2(o4), d2(o5), d3(o0), d3(o2), d3(o4), d3(o5), d3(o6), d4(o0), d4(o2), d4(o4), d4(o5), d4(o6), d5(o0), d5(o2), d5(o4), d5(o5), d5(o6), d8(o2), d8(o5), d8(o6)
}, {
  d0(o3).
  d1(o3).
  d2(o6).
  d6(o2).
  d6(o3).
  d7(o3).
  d7(o5).
  d8(o0).
  d8(o4).
}).