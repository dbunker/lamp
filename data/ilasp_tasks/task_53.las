#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o4).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d2(var(obj))).

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

#pos(eg1, {
  d2(o4)
}, {
  d2(o0), d2(o1), d2(o8), d2(o9)
}, {
  d0(o4).
  d1(o4).
  d1(o8).
  d3(o1).
  d4(o0).
  d4(o9).
}).