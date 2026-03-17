#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o2).
#constant(obj, o4).

#modeh(d1(var(obj))).
#modeh(d3(var(obj))).

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
  d1(o1), d1(o2), d1(o4), d3(o1), d3(o4)
}, {
  d1(o0), d3(o0), d3(o2)
}, {
  d0(o0).
  d2(o1).
  d2(o4).
  d4(o2).
}).