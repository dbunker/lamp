#constant(obj, o1).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o5).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).
#modeh(d4(var(obj))).
#modeh(d8(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).
#modeb(1, d9(var(obj))).
#modeb(1, d9(var(obj)), (negative)).

#pos(eg1, {
  d0(o1), d0(o5), d1(o1), d4(o1), d8(o3)
}, {
  d0(o3), d1(o3), d1(o4), d4(o3), d4(o4), d4(o5), d8(o1), d8(o4), d8(o5)
}, {
  d0(o4).
  d1(o5).
  d2(o3).
  d5(o1).
  d9(o1).
  d9(o4).
}).