#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o5).
#constant(obj, o6).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d2(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).

#pos(eg1, {
  d2(o2), d2(o6)
}, {
  d2(o1), d2(o8)
}, {
  d0(o1).
  d1(o8).
  d2(o5).
  d2(o9).
  d6(o2).
  d6(o6).
}).