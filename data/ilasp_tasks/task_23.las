#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).

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
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d5(o1)
}, {
  d5(o2), d5(o3), d5(o4), d5(o5)
}, {
  d0(o2).
  d1(o2).
  d2(o4).
  d3(o1).
  d4(o1).
  d4(o3).
  d6(o3).
  d7(o5).
  d8(o1).
}).