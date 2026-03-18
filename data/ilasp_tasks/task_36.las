#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).

#modeh(d2(var(obj))).
#modeh(d5(var(obj))).
#modeh(d6(var(obj))).
#modeh(d8(var(obj))).

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
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d2(o3), d5(o1), d5(o4), d6(o1), d6(o4), d8(o1), d8(o4)
}, {
  d2(o2), d2(o4), d5(o2), d5(o3), d6(o2), d8(o2), d8(o3)
}, {
  d1(o3).
  d2(o1).
  d3(o2).
  d4(o1).
  d4(o4).
  d6(o3).
}).