#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).
#constant(obj, o6).
#constant(obj, o8).

#modeh(d1(var(obj))).
#modeh(d2(var(obj))).
#modeh(d4(var(obj))).
#modeh(d5(var(obj))).
#modeh(d7(var(obj))).
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
  d1(o6), d2(o4), d2(o5), d2(o6), d2(o8), d4(o2), d4(o6), d5(o2), d7(o6), d8(o3)
}, {
  d1(o2), d1(o3), d1(o4), d1(o5), d1(o8), d2(o2), d2(o3), d4(o3), d5(o3), d5(o4), d5(o5), d5(o6), d5(o8), d7(o2), d7(o3), d7(o4), d7(o5), d7(o8), d8(o2), d8(o4), d8(o5), d8(o6), d8(o8)
}, {
  d0(o2).
  d3(o6).
  d4(o4).
  d4(o5).
  d4(o8).
  d6(o3).
}).