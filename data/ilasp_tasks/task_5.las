#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o6).
#constant(obj, o7).
#constant(obj, o8).

#modeh(d0(var(obj))).
#modeh(d4(var(obj))).
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
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d0(o6), d4(o6), d8(o6)
}, {
  d0(o2), d0(o3), d0(o4), d0(o7), d0(o8), d4(o2), d4(o3), d4(o4), d4(o7), d4(o8), d8(o2), d8(o3), d8(o4), d8(o7), d8(o8)
}, {
  d1(o4).
  d1(o6).
  d2(o7).
  d3(o2).
  d3(o6).
  d3(o8).
  d5(o3).
  d5(o6).
  d7(o6).
}).