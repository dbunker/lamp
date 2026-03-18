#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).

#modeh(d0(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).

#pos(eg1, {
  d0(o5)
}, {
  d0(o1), d0(o2), d0(o3), d0(o4)
}, {
  d1(o3).
  d3(o5).
  d4(o1).
  d5(o2).
  d6(o4).
  d6(o5).
}).